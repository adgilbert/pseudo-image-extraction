import logging
import os
import pickle
import time

import mesh_utils.utilities as utilities
from data_handlers.dataframes import VTKSliceDataframe, PseudoDatasetDataframe
from data_handlers.slice_editor import SliceHandler
from data_handlers.vtk_slicer import VTKSlicer
from export_config import BaseOptions


def get_pseudo_imgs_from_vtk(df, opts):
    pseudo_df = PseudoDatasetDataframe()
    for ind, row in df.iterrows():
        if row.model_id >= opts.max_models:
            break
        try:
            logging.info("==== processing image models {}, rotation {} ====".format(row.model_id, row.rotation_id))
            for pi_ind in range(opts.images_per_slice):
                slc = SliceHandler(row.vtk_img, opts)
                pi_dict = slc.create_pseudo_images(opts)
                pi_dict.update(row.to_dict().copy())
                pi_dict.update(img_df_num=pi_ind)
                pseudo_df.add_record(pi_dict)
        except AssertionError as e:
            logging.warning(f"{row.model_id}-{row.rotation_id} FAILED because {e}, skipping")
            print("My Check, CSD: ", row.model_id,"\t\t", row.rotation_id)
            continue
    return pseudo_df


if __name__ == "__main__":
    import numpy as np

    np.random.seed(21)
    opts = BaseOptions().parse()
    start_time = time.time()
    output_dir = os.path.join(opts.output_dir, opts.name)
    utilities.set_logger(os.path.join(output_dir, 'log.txt'))

    # VTK subsection
    if not opts.exclude_vtk:
        vtk_slice_df = VTKSliceDataframe(opts)
        vtk_slicer = VTKSlicer(opts)
        vtk_slicer.create_slices(vtk_slice_df)

        num_dfs = vtk_slice_df.df_id + 1  # + 1 b/c zero indexed
        vtk_time = time.time()
        samples = (num_dfs - 1) * vtk_slice_df.max_rows + vtk_slice_df.get_df().shape[0]
        logging.info(f'TIMER: VTK processing took {vtk_time - start_time:.2f} seconds for {samples} samples')

    # # Image subsection
    if not opts.exclude_image:
        img_start_time = time.time()
        num_dfs = len([f for f in os.listdir(output_dir) if 'vtk_df' in f and '.pck' in f])
        assert num_dfs > 0, f"no VTK dfs found in {output_dir}. rerun vtk subsection"
        for df_num in range(num_dfs):
            df_name = os.path.join(output_dir, f"vtk_df_{df_num}.pck")
            # if opts.verbose:
            logging.info(f'loading dataframe from {df_name}')
            df = pickle.load(open(df_name, 'rb'))
            if df.model_id.min() > opts.max_models:
                continue
            pseudo_df = get_pseudo_imgs_from_vtk(df, opts)
            img_df_name = os.path.join(output_dir, f"img_DF_{df_num}")
            logging.info(f"saving dataframe to {img_df_name}")
            pseudo_df.save_df(img_df_name + '.pck')
            try:
                pseudo_df.save_df(img_df_name + '.xlsx')
            except:
                logging.info("\tunable to save excel dataframe. Is file open?")
        img_end_time = time.time()
        logging.info(f'TIMER: Image processing took {img_end_time - img_start_time:.2f} seconds for {num_dfs} dfs')

    if not opts.exclude_export:
        num_dfs = len([f for f in os.listdir(output_dir) if 'img_DF' in f and '.pck' in f])
        assert num_dfs > 0, f"no image dfs found in {output_dir}. rerun vtk subsection"
        logging.info(f"Cleaning directory {os.path.join(output_dir, 'images')} and adding images to it")
        utilities.rmdir(os.path.join(output_dir, 'images'))  # remove directory
        for df_num in range(num_dfs):
            df_name = os.path.join(output_dir, f"img_DF_{df_num}.pck")
            logging.info(f'loading dataframe from {df_name}')
            pseudo_df = PseudoDatasetDataframe().restore_from_pickle(df_name)
            if opts.original_sized_images:
                pseudo_df.save_images_original_size(os.path.join(output_dir, "resized_images"), debug=False)
            else:
                pseudo_df.save_images(os.path.join(output_dir, "images"))
    end_time = time.time()
    logging.info('TIMER: full script took {:.2f} seconds'.format(end_time - start_time))
