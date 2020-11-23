import logging
import os
from pathlib import Path

import numpy as np
import vtk
from PIL import Image
from vtk.util.numpy_support import vtk_to_numpy


# 01. LV myocardium (endo + epi)
# 02. RV myocardium (endo + epi)
# 03. LA myocardium (endo + epi)
# 04. RA myocardium (endo + epi)
#
# 05. Aorta
# 06. Pulmonary artery
#
# 07. Mitral valve
# 08. Triscupid valve
#
# 09. Aortic valve
# 10. Pulmonary valve

# 11. Appendage
# 12. Left superior pulmonary vein
# 13. Left inferior pulmonary vein
# 14. Right inferior pulmonary vein
# 15. Right superior pulmonary vein
#
# 16. Superior vena cava
# 17. Inferior vena cava

# 18. Appendage border
# 19. Right inferior pulmonary vein border
# 20. Left inferior pulmonary vein border
# 21. Left superior pulmonary vein border
# 22. Right superior pulmonary vein border
# 23. Superior vena cava border
# 24. Inferior vena cava border



def calculate_rotation(reference_vector, target_vector):
    """
    Calculates the rotation matrix which rotates the object to align the target vector direction to reference
    vector direction. Assumes that both vectors are anchored at the beginning of the coordinate system
    :param reference_vector: Vector with referential direction. The rotation matrix will align the target_vector's
    direction to this one.
    :param target_vector:  Vector pointing to a  structure corresponding to the referential vector.
    :return: 3x3 rotation matrix (rot), where [rot @ target_vector = reference_vector] in terms of direction.
    """

    unit_reference_vector = reference_vector / np.linalg.norm(reference_vector)
    unit_target_vector = target_vector / np.linalg.norm(target_vector)
    c = unit_target_vector @ unit_reference_vector
    if c == 1:
        return np.eye(3)
    elif c == -1:
        return -np.eye(3)
    else:
        v = np.cross(unit_target_vector, unit_reference_vector)
        vx = np.array(([0,     -v[2],   v[1]],
                       [v[2],   0,     -v[0]],
                       [-v[1],  v[0],   0]))
        vx2 = vx @ vx
        return np.eye(3) + vx + vx2 / (1 + c)


class Heart:
    list_of_elements = ['LV', 'RV', 'LA', 'RA', 'AO', 'PA', 'MV', 'TV', 'AV', 'PV',
                        'APP', 'LSPV', 'LIPV', 'RIPV', 'RSPV', 'SVC', 'IVC',
                        'AB', 'RIPVB', 'LIPVB', 'LSPVB', 'RSPVB', 'SVCB', 'IVCB']

    # TODO: Add the dictionary with labels, to use in alignment

    def __init__(self, filename='h_case06.vtk', to_polydata=False, opts=None):

        self.filename, self.input_type = filename.split('.')
        # include opts without breaking backwards compatability
        if opts is not None:
            self.verbose=opts.verbose
            self.output_image_size = opts.image_size * 4
        else:
            logging.warning("WARNING: opts not used to initialize, using default values")
            self.verbose = True
            self.output_image_size = 1024
        # Write vtk output to a file
        w = vtk.vtkFileOutputWindow()
        w.SetFileName(str(Path(self.filename).parent / Path('errors.txt')))
        vtk.vtkOutputWindow.SetInstance(w)

        if self.verbose:
            logging.info(f"filename = {self.filename}")
            logging.info('Reading the data from {}.{}...'.format(self.filename, self.input_type))
        if self.input_type == 'obj':
            self.mesh, self.scalar_range = self.read_obj()
        elif self.input_type == 'vtp':
            self.mesh, self.scalar_range = self.read_vtp()
        else:
            self.mesh, self.scalar_range = self.read_vtk(to_polydata)

        # self.scalar_range = [1.0, 17.0]  # Added for particular case of CT meshes
        # print('Corrected scalar range: {}'.format(self.scalar_range))
        self.center_of_heart = self.get_center(self.mesh)
        if self.verbose:
            logging.info('Model centered at: {}'.format(self.center_of_heart))
        self.label = 0
        self._landmarks = dict()  # dict to contain found landmarks


    @staticmethod
    def get_center(_mesh):
        centerofmass = vtk.vtkCenterOfMass()
        centerofmass.SetInputData(_mesh.GetOutput())
        centerofmass.Update()
        return np.array(centerofmass.GetCenter())

    def visualize_mesh(self, display=True):
        # Create the mapper that corresponds the objects of the vtk file into graphics elements
        mapper = vtk.vtkDataSetMapper()
        try:
            mapper.SetInputData(self.mesh.GetOutput())
        except TypeError:
            print('Can\'t get output directly')
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(self.mesh.GetOutputPort())
        mapper.SetScalarRange(self.scalar_range)

        # Create the Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create the Renderer
        renderer = vtk.vtkRenderer()
        renderer.ResetCameraClippingRange()
        renderer.AddActor(actor)  # More actors can be added
        renderer.SetBackground(1, 1, 1)  # Set background to white

        # Create the RendererWindow
        renderer_window = vtk.vtkRenderWindow()
        renderer_window.AddRenderer(renderer)

        # Display the mesh
        # noinspection PyArgumentList
        if display:
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(renderer_window)
            interactor.Initialize()
            interactor.Start()
        else:
            return renderer_window

    # -----3D rigid transformations---------------------------------------------------------------------------

    def rotate(self, alpha=0, beta=0, gamma=0, rotation_matrix=None):
        rotate = vtk.vtkTransform()
        if rotation_matrix is not None:
            translation_matrix = np.eye(4)
            translation_matrix[:-1, :-1] = rotation_matrix
            if self.verbose:
                print('Translation matrix (rotation):\n', translation_matrix)
            rotate.SetMatrix(translation_matrix.ravel())
        else:
            rotate.Identity()
            rotate.RotateX(alpha)
            rotate.RotateY(beta)
            rotate.RotateZ(gamma)
        transformer = vtk.vtkTransformFilter()
        transformer.SetInputConnection(self.mesh.GetOutputPort())
        transformer.SetTransform(rotate)
        transformer.Update()
        self.mesh = transformer
        self.center_of_heart = self.get_center(self.mesh)
        # Invalidate landmarks after rotation
        self._landmarks = dict()

    def scale(self, factor=(0.001, 0.001, 0.001)):
        scale = vtk.vtkTransform()
        scale.Scale(factor[0], factor[1], factor[2])
        transformer = vtk.vtkTransformFilter()
        transformer.SetInputConnection(self.mesh.GetOutputPort())
        transformer.SetTransform(scale)
        transformer.Update()
        self.mesh = transformer
        self.center_of_heart = self.get_center(self.mesh)
        if self.verbose:
            logging.info(self.center_of_heart)
        # Invalidate landmarks after scale
        self._landmarks = dict()

    def translate(self, rotation_matrix, translation_vector):
        translate = vtk.vtkTransform()
        translation_matrix = np.eye(4)
        translation_matrix[:-1, :-1] = rotation_matrix
        translation_matrix[:-1, -1] = translation_vector
        if self.verbose:
            print('Translation matrix:\n', translation_matrix)
        translate.SetMatrix(translation_matrix.ravel())
        transformer = vtk.vtkTransformFilter()
        transformer.SetInputConnection(self.mesh.GetOutputPort())
        transformer.SetTransform(translate)
        transformer.Update()
        self.mesh = transformer
        self.center_of_heart = self.get_center(self.mesh)
        # Invalidate landmarks after translate
        self._landmarks = dict()

    def translate_to_center(self, label=None):
        # vtkTransform.SetMatrix - enables for applying 4x4 transformation matrix to the meshes
        # if label is provided, translates to the center of the element with that label
        translate = vtk.vtkTransform()
        if self.verbose:
            logging.info('translating_to_center')
        if label is not None:
            central_element = self.threshold(label, label)
            center_of_element = self.get_center(central_element)
            translate.Translate(-center_of_element[0], -center_of_element[1], -center_of_element[2])
        else:
            translate.Translate(-self.center_of_heart[0], -self.center_of_heart[1], -self.center_of_heart[2])
        translate.Update()
        transformer = vtk.vtkTransformFilter()
        transformer.SetInputConnection(self.mesh.GetOutputPort())
        transformer.SetTransform(translate)
        transformer.Update()
        self.mesh = transformer
        self.center_of_heart = self.get_center(self.mesh)
        if self.verbose:
            logging.info(self.center_of_heart)
        # Invalidate landmarks after translate
        self._landmarks = dict()

    # -----Mesh manipulation----------------------------------------------------------------------------------
    def align_slice(self, a, b, c, preferred_direction=None):
        """ align a slice to the given plane given 3 landmarks. If preferred_direction is given then this function
        will flip the norm to choose the one that was closest to the preferred_direction """
        def _choose_closest(x, ys):
            "choose the y that is closest to x (by norm) "
            min_dist = None
            chosen_y = None
            assert len(ys) > 1, "must give at least one y to choose closest"
            for y in ys:
                dist = np.linalg.norm(x-y)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    chosen_y = y
            return chosen_y
        center = np.mean((a, b, c), axis=0)
        self.translate(rotation_matrix=np.eye(3), translation_vector=-center)

        a2, b2, c2 = [x - center for x in [a, b, c]]
        _normal = calculate_plane_normal(a2, b2, c2)
        if preferred_direction is not None:
            reverse_normal = _normal * -1
            _normal = _choose_closest(preferred_direction, [_normal, reverse_normal])
        rot1 = calculate_rotation(np.array([0, 0, 1]), _normal)
        a3, b3, c3 = [rot1 @ x for x in [a2, b2, c2]]
        rot2 = calculate_rotation(np.array([0, 1, 0]), b3 / np.linalg.norm(b3))
        rot3 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        rot = rot3 @ rot2 @ rot1
        self.rotate(rotation_matrix=rot)
        # Invalidate landmarks after align
        self._landmarks = dict()
        return _normal

    def apply_modes(self, modes_with_scales):
        for mode, scale in modes_with_scales.items():
            if self.verbose:
                logging.info('Applying ' + mode + ' multiplied by ' + str(scale))
            self.mesh.GetOutput().GetPointData().SetActiveVectors(mode)
            warp_vector = vtk.vtkWarpVector()
            warp_vector.SetInputConnection(self.mesh.GetOutputPort())
            warp_vector.SetScaleFactor(scale)
            warp_vector.Update()
            self.mesh = warp_vector
        # Invalidate landmarks after apply_modes
        self._landmarks = dict()

    def build_tag(self, label):
        self.label = label
        tag = vtk.vtkIdFilter()
        tag.CellIdsOn()
        tag.PointIdsOff()
        tag.SetInputConnection(self.mesh.GetOutputPort())
        tag.SetIdsArrayName('elemTag')
        tag.Update()
        self.mesh = tag

    def change_tag_label(self):
        size = self.mesh.GetOutput().GetAttributes(1).GetArray(0).GetSize()
        for id in range(size):
            self.mesh.GetOutput().GetAttributes(1).GetArray(0).SetTuple(id, (float(self.label),))

    def clean_polydata(self, tolerance=0.005, remove_lines=False):
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(self.mesh.GetOutputPort())
        cleaner.SetTolerance(tolerance)
        cleaner.ConvertLinesToPointsOn()
        cleaner.ConvertPolysToLinesOn()
        cleaner.ConvertStripsToPolysOn()
        cleaner.Update()
        self.mesh = cleaner
        if remove_lines:
            self.mesh.GetOutput().SetLines(vtk.vtkCellArray())

    def decimation(self, reduction=50):
        decimation = vtk.vtkQuadricDecimation()
        decimation.SetInputConnection(self.mesh.GetOutputPort())
        decimation.VolumePreservationOn()
        decimation.SetTargetReduction(reduction / 100)  # percent of removed triangles
        decimation.Update()
        self.mesh = decimation

    def delaunay2d(self):
        delaunay2d = vtk.vtkDelaunay2D()
        delaunay2d.SetInputConnection(self.mesh.GetOutputPort())
        delaunay2d.Update()
        self.mesh = delaunay2d

    def delaunay3d(self):
        delaunay3d = vtk.vtkDelaunay3D()
        delaunay3d.SetInputConnection(self.mesh.GetOutputPort())
        delaunay3d.Update()
        self.mesh = delaunay3d

    def extract_surface(self):
        # Get surface of the mesh
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(self.mesh.GetOutput())
        surface_filter.Update()
        self.mesh = surface_filter

    def fill_holes(self, hole_size=10.0):
        filling_filter = vtk.vtkFillHolesFilter()
        filling_filter.SetInputConnection(self.mesh.GetOutputPort())
        filling_filter.SetHoleSize(hole_size)
        filling_filter.Update()
        self.mesh = filling_filter

    @staticmethod
    def get_external_surface(_mesh, external=True):
        _center = np.zeros(3)
        _bounds = np.zeros(6)
        _ray_start = np.zeros(3)
        cell_id = vtk.mutable(-1)
        xyz = np.zeros(3)
        pcoords = np.zeros(3)
        t = vtk.mutable(0)
        sub_id = vtk.mutable(0)
        if external:
            surf = 1.1
        else:
            surf = -1.1

        _mesh.GetOutput().GetCenter(_center)
        _mesh.GetOutput().GetPoints().GetBounds(_bounds)
        for j in range(3):
            _ray_start[j] = _bounds[2 * j + 1] * surf

        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(_mesh.GetOutput())
        cell_locator.BuildLocator()
        cell_locator.IntersectWithLine(_ray_start, _center, 0.0001, t, xyz, pcoords, sub_id, cell_id)
        logging.info('ID of the cell on the outer surface: {}'.format(cell_id))

        connectivity_filter = vtk.vtkConnectivityFilter()
        connectivity_filter.SetInputConnection(_mesh.GetOutputPort())
        connectivity_filter.SetExtractionModeToCellSeededRegions()
        connectivity_filter.InitializeSeedList()
        connectivity_filter.AddSeed(cell_id)
        connectivity_filter.Update()
        return connectivity_filter

    def measure_average_edge_length(self):
        size = vtk.vtkCellSizeFilter()
        size.SetInputConnection(self.mesh.GetOutputPort())
        size.Update()

    def normals(self):
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(self.mesh.GetOutputPort())
        normals.FlipNormalsOn()
        normals.Update()
        self.mesh = normals

    def resample_to_image(self, label_name='elemTag'):

        resampler = vtk.vtkResampleToImage()
        resampler.SetInputConnection(self.mesh.GetOutputPort())
        resampler.UseInputBoundsOff()
        bounds = np.array(self.mesh.GetOutput().GetBounds())
        bounds[:4] = bounds[:4] + 0.1 * bounds[:4]
        assert np.sum(bounds[4:] < 0.001), 'The provided slice must be 2D and must be projected on the XY plane'

        resampler.SetSamplingBounds(*bounds[:5], 1.01)
        resampler.SetSamplingDimensions(self.output_image_size, self.output_image_size, 1)
        resampler.Update()

        img_as_array = vtk_to_numpy(resampler.GetOutput().GetPointData().GetArray(label_name))
        img_as_array = img_as_array.reshape((int(np.sqrt(img_as_array.shape[0])), int(np.sqrt(img_as_array.shape[0]))))

        return img_as_array

    def slice_extraction(self, origin, normal):
        # create a plane to cut (xz normal=(1,0,0);XY =(0,0,1),YZ =(0,1,0)
        plane = vtk.vtkPlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)

        # create cutter
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputConnection(self.mesh.GetOutputPort())
        cutter.Update()

        self.mesh = cutter

    def smooth_laplacian(self, number_of_iterations=50):
        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(self.mesh.GetOutputPort())
        smooth.SetNumberOfIterations(number_of_iterations)
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOn()
        smooth.Update()
        self.mesh = smooth

    def smooth_window(self, number_of_iterations=15, pass_band=0.5):
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputConnection(self.mesh.GetOutputPort())
        smooth.SetNumberOfIterations(number_of_iterations)
        smooth.BoundarySmoothingOn()
        smooth.FeatureEdgeSmoothingOff()
        smooth.SetPassBand(pass_band)
        smooth.NonManifoldSmoothingOn()
        smooth.NormalizeCoordinatesOn()
        smooth.Update()
        self.mesh = smooth

    def subdivision(self, number_of_subdivisions=3):
        self.normals()
        subdivision = vtk.vtkLinearSubdivisionFilter()
        subdivision.SetNumberOfSubdivisions(number_of_subdivisions)
        subdivision.SetInputConnection(self.mesh.GetOutputPort())
        subdivision.Update()
        self.mesh = subdivision
        self.visualize_mesh(True)

    def tetrahedralize(self, leave_tetra_only=True):
        tetra = vtk.vtkDataSetTriangleFilter()
        if leave_tetra_only:
            tetra.TetrahedraOnlyOn()
        tetra.SetInputConnection(self.mesh.GetOutputPort())
        tetra.Update()
        self.mesh = tetra

    def threshold(self, low=0, high=100):
        threshold = vtk.vtkThreshold()
        threshold.SetInputConnection(self.mesh.GetOutputPort())
        threshold.ThresholdBetween(low, high)
        threshold.Update()
        # choose scalars???
        return threshold

    def ug_geometry(self):
        geometry = vtk.vtkUnstructuredGridGeometryFilter()
        if self.verbose:
            print(geometry.GetDuplicateGhostCellClipping())
        geometry.SetInputConnection(self.mesh.GetOutputPort())
        geometry.Update()
        self.mesh = geometry

    def unstructured_grid_to_poly_data(self):
        geometry_filter = vtk.vtkExtractGeometry()
        geometry_filter.SetInputConnection(self.mesh.GetOutputPort())
        geometry_filter.Update()
        return geometry_filter

    # -----MeshInformation------------------------------------------------------------------------------------
    def get_volume(self):
        mass = vtk.vtkMassProperties()
        mass.SetInputConnection(self.mesh.GetOutputPort())
        return mass.GetVolume()

    def print_numbers(self):
        _mesh = self.mesh.GetOutput()
        print('Number of verices: {}'.format(_mesh.GetNumberOfVerts()))
        print('Number of lines: {}'.format(_mesh.GetNumberOfLines()))
        print('Number of strips: {}'.format(_mesh.GetNumberOfStrips()))
        print('Number of polys: {}'.format(_mesh.GetNumberOfPolys()))
        print('Number of cells: {}'.format(_mesh.GetNumberOfCells()))
        print('Number of points: {}'.format(_mesh.GetNumberOfPoints()))

    # -----InputOutput----------------------------------------------------------------------------------------

    # -----Readers

    def read_vtk(self, to_polydata=False):
        # Read the source file.
        assert os.path.isfile('.'.join([self.filename, self.input_type])), \
            'File {} does not exist!'.format('.'.join([self.filename, self.input_type]))
        reader = vtk.vtkDataReader()
        reader.SetFileName('.'.join([self.filename, self.input_type]))
        reader.Update()
        logging.info('Case ID : {}, input type: {}'.format(self.filename, self.input_type))
        if reader.IsFileUnstructuredGrid():
            logging.info('Reading Unstructured Grid...')
            reader = vtk.vtkUnstructuredGridReader()
        elif reader.IsFilePolyData():
            logging.info('Reading Polygonal Mesh...')
            reader = vtk.vtkPolyDataReader()
        elif reader.IsFileStructuredGrid():
            logging.info('Reading Structured Grid...')
            reader = vtk.vtkStructuredGridReader()
        elif reader.IsFileStructuredPoints():
            logging.info('Reading Structured Points...')
            reader = vtk.vtkStructuredPointsReader()
        elif reader.IsFileRectilinearGrid():
            logging.info('Reading Rectilinear Grid...')
            reader = vtk.vtkRectilinearGridReader()
        else:
            logging.info('Data format unknown...')
        reader.SetFileName(self.filename + '.' + self.input_type)
        reader.Update()  # Needed because of GetScalarRange
        scalar_range = reader.GetOutput().GetScalarRange()
        if to_polydata and not reader.IsFilePolyData():
            logging.info('Transform to Polygonal Mesh')
            reader = self.unstructured_grid_to_poly_data() # reader)
        if self.verbose:
            logging.info('Scalar range: \n{}'.format(scalar_range))
        return reader, scalar_range

    def read_vtp(self):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName('.'.join([self.filename, self.input_type]))
        reader.Update()
        scalar_range = reader.GetOutput().GetScalarRange()
        return reader, scalar_range

    def read_obj(self):
        reader = vtk.vtkOBJReader()
        reader.SetFileName('.'.join([self.filename, self.input_type]))
        reader.Update()
        scalar_range = reader.GetOutput().GetScalarRange()
        return reader, scalar_range

    # -----Writers

    def write_mha(self):

        output_filename = self.filename + '.mha'
        # output_filename_raw = self.filename + '.raw'
        logging.info('writing mha')

        mha_writer = vtk.vtkMetaImageWriter()
        mha_writer.SetInputConnection(self.mesh.GetOutputPort())
        mha_writer.SetFileName(output_filename)
        # mha_writer.SetRAWFileName(output_filename_raw)
        mha_writer.Write()

    def write_stl(self):
        output_filename = self.filename + '.stl'

        # Get surface of the mesh
        logging.info('Extracting surface to save as .STL file...')
        # self.extract_surface()

        # Write file to .stl format
        stl_writer = vtk.vtkSTLWriter()
        stl_writer.SetFileName(output_filename)
        stl_writer.SetInputConnection(self.mesh.GetOutputPort())
        stl_writer.Write()
        logging.info('{} written succesfully'.format(output_filename))

    def write_obj(self, postscript=''):
        output_filename = self.filename
        render_window = self.visualize_mesh(False)

        logging.info('Saving PolyData in the OBJ file...')
        obj_writer = vtk.vtkOBJExporter()
        obj_writer.SetRenderWindow(render_window)
        obj_writer.SetFilePrefix(output_filename + postscript)
        obj_writer.Write()
        logging.info('{} written succesfully'.format(output_filename + postscript + '.obj'))

    def write_png(self, postscript=''):

        logging.info('Saving slice in PNG file...')
        output_filename = self.filename + postscript + '.png'
        image = Image.fromarray(self.resample_to_image().astype(np.uint8), 'L')
        image = image.convert('L')
        image.save(output_filename, 'PNG')
        logging.info('{} written succesfully'.format(output_filename))

    def get_PIL(self):
        image = Image.fromarray(self.resample_to_image().astype(np.uint8), 'L')
        image = image.convert('L')
        return image

    def write_vtk(self, postscript='_new', type_='PolyData', outname=None):
        """ outname overwrites self.filename if given"""
        fname = outname if outname is not None else self.filename
        output_filename = fname + postscript + '.vtk'
        writer = None
        if type_ == 'PolyData':
            logging.info('Saving PolyData...')
            self.extract_surface()
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputConnection(self.mesh.GetOutputPort())
        elif type_ == 'UG':
            logging.info('Saving Unstructured Grid...')
            writer = vtk.vtkUnstructuredGridWriter()
            appendFilter = vtk.vtkAppendPolyData()
            appendFilter.AddInputData(self.mesh.GetOutput())
            appendFilter.Update()
            writer.SetInputConnection(appendFilter.GetOutputPort())
        else:
            exit("Select \'Polydata\' or \'UG\' as type of the saved mesh")

        writer.SetFileName(output_filename)
        writer.Update()
        writer.Write()
        logging.info('{} written succesfully'.format(output_filename))

    def write_vtk_points(self, postscript='_points'):
        output_filename = self.filename + postscript + '.vtk'

        point_cloud = vtk.vtkPolyData()
        point_cloud.SetPoints(self.mesh.GetOutput().GetPoints())
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(point_cloud)
        writer.SetFileName(output_filename)
        writer.Update()
        writer.Write()
    # --------------------


def calculate_plane_normal(a, b, c):
    """
    :param a: 3D point
    :param b: 3D point
    :param c: 3D point
    :return: Vector normal to a plane which crosses the abc points.
    """
    x = np.cross(b-a, b-c)
    return x/np.linalg.norm(x)