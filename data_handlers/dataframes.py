import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from box import Box

import mesh_utils.image_operations as image_operations
import mesh_utils.utilities as utilities
from mesh_utils import Tags
from mesh_utils.image_landmarks import a4ch_image_landmarks


class VTKSliceDataframe(object):
    """ class designed to keep track of the exported data """

    def __init__(self, opts):
        self._df = pd.DataFrame()
        self.required_keys = ['source_file', 'rotation', 'origin', 'landmarks', 'aligned', 'dest_file', 'vtk_img',
                              'model_id']
        self.vtk_id = 0
        self.df_id = 0
        self.filename = os.path.join(opts.output_dir, opts.name, "vtk_df")
        self.max_rows = 60  # save images and df after this many rows
        self.include_inverse_images = opts.include_inverse_images
        self.inverses_included = False
        self.verbose = opts.verbose

    def load_from_pck(self, filename):
        """ Load dataframe from pickle file
        :param filename: Full path to the pickle file
        """
        self._df = pickle.load(open(filename, 'rb'))
        self.vtk_id = np.max(self._df.vtk_id)

    def add_record(self, record: dict):
        """ Add a row to the dataframe. Will save dataframe if max rows is reached (avoids disk errors)
        :param record: a dictionary with each key representing a column in the dataframe
        """
        for r in self.required_keys:
            assert r in record, "record is missing required key {}".format(r)
        # assign a unique id
        record['vtk_id'] = self.vtk_id
        # assign dir column to simplify looking at where it came from
        record['dir'] = os.path.dirname(record['source_file'])
        self.vtk_id += 1
        self._df = self._df.append(record, ignore_index=True)
        # if array gets too large than it will fail so save periodically instead
        if self._df.shape[0] >= self.max_rows:
            print("reached max rows, saving and starting a new df")
            self.save_df()
            self._df = pd.DataFrame()
            self.df_id += 1
            self.inverses_included = False

    def get_df(self):
        return self._df

    def save_df(self, exts=(".pck", ".xlsx")):
        """  Save the current dataframe. Can either save as a pickle file or an excel file
        :param exts: how to save dataframe. Default saves both pickle and excel.
        """
        if self._df.shape[0] < 1:
            print("no rows in df, not saving...")
            return
        if self.include_inverse_images:
            self._add_inverse_images()  # TODO: need to handle axis name parameter here somehow
        for ext in exts:
            filename = self.filename + '_' + str(self.df_id) + ext
            if ext == '.pck':
                # export a pandas dataframe
                pickle.dump(self._df, open(filename, 'wb'))
            elif ext == '.xlsx':
                cols = list(self._df.columns)
                cols.remove('vtk_img')  # don't save image data to excel
                if self.verbose:
                    logging.info(f"Dataframe cols in excel: {cols}")
                try:
                    self._df.to_excel(filename, columns=cols)
                except:
                    logging.info("\tunable to save excel dataframe. Is file open?")
            elif ext == '.heart':
                # format to load back into this class
                pickle.dump(self, open(filename, 'wb'))
            else:
                raise ValueError("file type {} not among recognized options".format(ext))

    def recreate_vtk(self, output_dir):
        """ rewrite all vtk files in the dataframe """
        from .heart import Heart
        for i, row in self._df.iterrows():
            heart = Heart(row.source_file)
            outname = os.path.join(output_dir, str(row.unique_id))
            heart.write_vtk('', type_='PolyData', outname=outname)

    def _add_inverse_images(self, axis="axis0"):
        """ add the inverse of all images """
        logging.warning("Adding inverse images but assuming full rotation so last is dropped. See code for more.")
        # The assumption in this function is that if you are flipping images it's because you want the entire rotation
        # Therefore this function will actually overwrite the last image in each rotation since that image is actually
        # just the flipped version of the first image.
        flip_df = self._df[~self._df.has_inverse]  # only include for rows where inverse hasn't been addded
        num_rotations = flip_df.rotation_id.max()
        new_rows = list()
        for i, row in flip_df.iterrows():
            row_copy = row.copy()
            row_copy.rotation_id += num_rotations
            row_copy[axis] = utilities.check_angle(row.axis0 + np.pi)
            row_copy.vtk_img = row_copy.vtk_img.transpose(Image.FLIP_TOP_BOTTOM)  # actually flips left to right
            new_rows.append(row_copy)
        # separate out adding new_rows to avoid changing size of df during iteration
        for nr in new_rows:
            self._df = self._df.append(nr, ignore_index=True)
            # drop the one duplicate created in the middle when start gets rotated
        self._df.drop_duplicates(["model_id", "rotation_id"], inplace=True)
        # drop the last value because this one is duplicate of first
        num_rotations = self._df.rotation_id.max()
        self._df = self._df[(self._df.rotation_id != num_rotations) | self._df.has_inverse]  # don't remove previous samples
        self._df.sort_values(["model_id", "rotation_id"], inplace=True)
        self._df["has_inverse"] = True  # mark that all rows have an inverse

    def save_images(self, output_dir):
        """ save the vtk images as pngs to output_dir """
        utilities.mkdir(os.path.join(output_dir, 'slices'))
        for _, row in self._df.iterrows():
            save_name = '_'.join(['models{:d}'.format(int(row.model_id)),
                                  'v{:d}'.format(int(row.vtk_id)),
                                  row.dir + '.png'])
            slice_save_name = os.path.join(output_dir, 'slices', save_name)
            img = image_operations.process_output_image(row.vtk_img)
            image_operations.save_img(img, slice_save_name)


class PseudoDatasetDataframe(object):
    """ class designed to keep track of the exported data """
    def __init__(self, unique_id=0):
        self._df = pd.DataFrame()
        self.required_keys = ['tissue_img', 'label_img', 'vtk_id', 'img_df_num', 'source_file', 'dir']
        self.unique_id = unique_id  # can be passed in to continue numbering from a previous dataframe

    def restore_from_pickle(self, df_filename, overwrite_unique_id=False):
        """ restore a class of this from a saved dataframe"""
        df = pickle.load(open(df_filename, 'rb'))
        for r in self.required_keys:
            assert r in list(df.columns), f"required key {r} is missing from  {df_filename} with cols {df.columns}"
        self._df = df
        if overwrite_unique_id:
            num_rows = df.shape[0]
            new_ids = np.arange(self.unique_id, self.unique_id + num_rows)
            self._df.unique_id = new_ids
        self.unique_id = self._df.unique_id.max() + 1
        return self

    def add_record(self, record: dict):
        for r in self.required_keys:
            assert r in record, "record is missing required key {}".format(r)
        # assign a unique id
        if "unique_id" not in record:
            record['unique_id'] = self.unique_id
            self.unique_id += 1
        # assign dir column to simplify looking at where it came frome
        self._df = self._df.append(record, ignore_index=True)

    def get_df(self):
        return self._df

    def __len__(self):
        return self._df.shape[0]

    def save_df(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.pck':
            # export a pandas dataframe
            pickle.dump(self._df, open(filename, 'wb'))
        elif ext == '.xlsx':
            cols = list(self._df.columns)
            cols.remove('label_img')
            cols.remove('tissue_img')
            cols.remove('vtk_img')
            cols.remove("cone_img")
            cols.remove("normal_img")
            # don't save image data to excel
            self._df.to_excel(filename, columns=cols)
        else:
            raise ValueError("filetype {} not among recognized options".format(ext))

    def save_images_original_size(self, output_dir, debug=False):
        """ save_images, but first resize to the original size by finding the original landmarks again """
        utilities.mkdirs(output_dir)
        utilities.mkdirs(os.path.join(output_dir, "labels"))
        if debug:
            utilities.mkdirs(os.path.join(output_dir, "plotted_landmarks"))
        bbox_lims = [0, 0, 0, 0]
        images = dict()
        original_distances = dict()
        new_distances = dict()
        for _, row in self._df.iterrows():
            save_name = '_'.join(['u{:03d}'.format(int(row.unique_id)),
                                  'model{:03d}'.format(int(row.model_id)),
                                  'r{:03d}'.format(int(row.rotation_id)),
                                  'v{:03d}'.format(int(row.vtk_id)),
                                  'i{:03d}'.format(int(row.img_df_num)),
                                  Path(row.dir).parts[-1] + '.png'])
            # New landmarks are the landmarks in the image. Old landmarks are the landmarks in the model
            new_landmarks = a4ch_image_landmarks(row.label_img)
            old_landmarks = row.landmarks

            # Project the old landmarks to the 2D plane of the image. First find the plane
            dx, dy, origin = row.rotation["dir_x"], row.rotation["dir_y"], row.rotation["origin"]
            normal = np.cross(dx, dy)
            normal /= np.linalg.norm(normal)
            dx /= np.linalg.norm(dx)
            dy = np.cross(dx, normal)  # original dy not guaranteed to be orthogonal to dx
            # Our plane is now defined by dx, dy and origin. Normal is redundant.
            # origin doesn't really matter since we later remove the shift component of the affine transform
            shifted_landmarks = [utilities.project3d_to_2d(o, dx, dy, origin, normal) for o in old_landmarks]
            if debug:
                import matplotlib.pyplot as plt
                # plt.imshow(row.label_img)
                for l, name, symbol, in zip(new_landmarks, ["mv", "tv", "apex"], ["o", "+", "*"]):
                    plt.plot(l[1], l[0], symbol, color="r", label=name)
                for l, symbol, in zip(shifted_landmarks, ["o", "+", "*"]):
                    plt.plot(l[1], l[0], symbol, color="b")
                plt.legend()
                plt.savefig(os.path.join(output_dir, "plotted_landmarks", save_name))
                plt.close()

            affine = utilities.solve_affine(shifted_landmarks, new_landmarks)
            # Get only the scale component
            scale = utilities.separate_affine(affine)

            # add a margin to the image. Otherwie the features will go outside the borders.
            padded_image = utilities.add_margin(row.label_img, 10, 134, 134, 10, color=Tags["label_background"])
            new_image = padded_image.transform((400, 400), Image.AFFINE, scale.ravel()[0:6], resample=Image.NEAREST,
                                               fillcolor=Tags["label_background"])

            post_landmarks = a4ch_image_landmarks(new_image)
            original_distances[save_name] = [np.linalg.norm(shifted_landmarks[p[0]] - shifted_landmarks[p[1]]) for p in
                                             [[0, 1], [0, 2], [1, 2]]]
            new_distances[save_name] = [np.linalg.norm(post_landmarks[p[0]] - post_landmarks[p[1]]) for p in
                                        [[0, 1], [0, 2], [1, 2]]]

            # find bounding box of important regions (global for all images)
            non_background = Image.fromarray((np.array(new_image) != Tags["label_background"]))
            bbox = non_background.getbbox()
            for i in range(4):
                if bbox_lims[i] == 0 or (i < 2 and bbox[i] < bbox_lims[i]) or (i >= 2 and bbox[i] > bbox_lims[i]):
                    bbox_lims[i] = bbox[i]
            images[save_name] = new_image
            if debug:
                image_operations.save_img(row.label_img,
                                          os.path.join(output_dir, "plotted_landmarks", "orig_" + save_name))
                image_operations.save_img(new_image, os.path.join(output_dir, "plotted_landmarks", "new_" + save_name))

        # compare the distance of the original vtk landmarks to the distances in the new_image.
        # the ratio between images should be the same
        orig0 = next(iter(original_distances.values()))
        new0 = next(iter(new_distances.values()))
        for o, n in zip(original_distances.values(), new_distances.values()):
            print(f"original: {np.array(o) / np.array(orig0)}")
            print(f"new: {np.array(n) / np.array(new0)}")

        # make it so image is not right at borders
        bbox_lims[0] = max(bbox_lims[0] - 10, 0)
        bbox_lims[1] = max(bbox_lims[1] - 10, 0)
        bbox_lims[2] = min(bbox_lims[2] + 10, 400)
        bbox_lims[3] = min(bbox_lims[3] + 10, 400)
        print(f"bbox_lims = {bbox_lims} ({bbox_lims[2] - bbox_lims[0]}, {bbox_lims[3] - bbox_lims[1]})=> (256, 256)")
        for save_name, im in images.items():
            im = im.crop(bbox_lims)
            # im = im.resize((256, 256), resample=Image.NEAREST)
            im = utilities.shift_to_center(im, Tags["label_background"])
            im, _ = image_operations.process_output_image(im, include_blood=True)
            image_operations.save_img(im, os.path.join(output_dir, "labels", save_name))

    def save_images(self, output_dir):
        utilities.mkdirs(output_dir)
        utilities.mkdirs(os.path.join(output_dir, 'pseudos'))
        utilities.mkdirs(os.path.join(output_dir, 'labels'))
        utilities.mkdirs(os.path.join(output_dir, 'normals'))
        utilities.mkdirs(os.path.join(output_dir, 'cones'))
        utilities.mkdirs(os.path.join(output_dir, "metas"))
        for _, row in self._df.iterrows():
            save_name = '_'.join(['u{:03d}'.format(int(row.unique_id)),
                                  'model{:03d}'.format(int(row.model_id)),
                                  'r{:03d}'.format(int(row.rotation_id)),
                                  'v{:03d}'.format(int(row.vtk_id)),
                                  'i{:03d}'.format(int(row.img_df_num)),
                                  Path(row.dir).parts[-1] + '.png'])
            tissue_save_name = os.path.join(output_dir, 'pseudos', save_name)
            label_save_name = os.path.join(output_dir, 'labels', save_name)
            norm_save_name = os.path.join(output_dir, 'normals', save_name)
            cone_save_name = os.path.join(output_dir, 'cones', save_name)
            meta_save_name = os.path.join(output_dir, "metas", save_name)
            label_img, label_vals = image_operations.process_output_image(row.label_img, include_blood=True)
            image_operations.save_img(row.tissue_img, tissue_save_name)
            image_operations.save_img(row.normal_img, norm_save_name)
            image_operations.save_img(label_img, label_save_name)
            image_operations.save_img(255. * np.array(row.cone_img), cone_save_name)
            json.dump(row.transform_params, open(meta_save_name, 'w'))
            # meta = row.to_dict()
            # meta = {k: v for k, v in meta.items() if all([i not in k for i in ["_img", "landmarks", "origin", "rotation", "cone_params"]])}
            # meta = Box(utilities.clean_dict_for_json(meta))
            # meta.to_json(meta_save_name)
        label_path = os.path.join(output_dir, "labels.json")
        Box(label_vals).to_json(label_path)

    def _show_img(self, img):
        """ helper function for debugging """
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()
