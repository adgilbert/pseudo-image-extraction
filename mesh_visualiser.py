import numpy as np
import pyvista as pv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from random import sample


class MeshVisualizer:
    '''
    Class that provides functionality for generating different views of the heart models 
    as well as its derivatives
    '''

    def __init__(self, sliceDir,  dataDir= Path("data/runData"), csvName="vtk_df_0.xlsx"):
        self.sliceDir = Path(sliceDir)
        self.dataDir = Path(dataDir)
        self.csvName = csvName

        self.slicesPerModel = None
        self.slices = self._sliceMaker()

        csvPath = self.sliceDir / self.csvName
        self.df = pd.read_excel(csvPath, engine='openpyxl')


    def _readAsDict(self, x): return eval(
        x.replace("array(", "").replace(")", ""))

    def _readAsArray(self, x):
        return x.replace("array(", "").replace(")", "")

    def _getFiles(self, dirPath):
        '''
            Returns the paths to the files in the specified directory. Ignores subdirectories
        '''
        return [p for p in Path(dirPath).iterdir() if p.is_file()]

    def _readMesh(self, path: list):
        '''
            Read the .vtk files and returns a list of meshes
        '''
        return [pv.read(p) for p in path]

    def _genModelPaths(self, modelNo: tuple):
        '''
            Constructs a list of paths to models based on model number
        '''
        return [Path("data/runData/Full_Heart_w_peri_{}.vtk".format(n)) for n in modelNo]

    def _normalise(self, x):
        '''
            Returns a normalised form of the input vector x
        '''
        return x/np.linalg.norm(x)

    def _sliceMaker(self):
        '''
            Create a dictionary with keys corresponding to model number
            and value containing all slices available for that model
        '''
        slicePaths = sorted(self._getFiles(self.sliceDir / "vtk"))
        self.slicesPerModel = int(
            str(slicePaths[-1]).split("_")[-1].split(".")[0]) + 1
        numHearts = len(slicePaths)/self.slicesPerModel
        assert not(len(slicePaths) % self.slicesPerModel)

        slices = {}
        for i in range(int(numHearts)):
            slices[i] = self._readMesh(slicePaths[3*i:3*(i + 1)])
        return slices

    def _getOriginAndNormal(self, modelNo):
        '''
            Returns nested lists containing origins and normals. For each model, there are
            several slices hence the nesting. The outer dimension
            changes with model whilst the inner lists contain data for each model's slices
        '''
        csvPath = self.sliceDir / self.csvName
        df = pd.read_excel(csvPath, engine='openpyxl')
        df["rotation"] = df["rotation"].apply(self._readAsDict)
        desiredModelPaths = self._genModelPaths(modelNo)

        normals, origins = [], []
        for p in desiredModelPaths:
            sliceRows = df.loc[df["source_file"] == str(p)]

            sliceNormals = [self._normalise(
                np.cross(sR["dir_x"], sR["dir_y"])) for sR in sliceRows["rotation"]]
            sliceOrigins = [sR["origin"] for sR in sliceRows["rotation"]]
            normals.append(sliceNormals)
            origins.append(sliceOrigins)

        return origins, normals

    def _getLandmarks(self, modelNo):
        temp = self.df.copy()
        temp["landmarks"] = temp["landmarks"].apply(self._readAsArray)

        landmarks = {}
        paths = self._genModelPaths(modelNo)
        for i, p in enumerate(paths):
            sliceRows = temp.loc[temp["source_file"] == str(p)]
            landmarks[modelNo[i]] = [np.array(eval(x)) for x in sliceRows["landmarks"]]

        return landmarks

    def _sphereMaker(self, modelNo):
        landmarksDict = self._getLandmarks(modelNo)
        self.landmarks = {m : [[pv.Sphere(radius=2, center=point)for point in slyce] for slyce in landmarksDict[m]] for m in modelNo}
        return 


    def _clipMaker(self, modelNo=(0,)):
        '''
            Create a dictionary with keys corresponding to model number and value containing all 
            clips available for that model. For each model, the clips are generated using the normals
            and origins of slice planes
        '''
        models = self._readMesh(self._genModelPaths(modelNo))
        origins, normals = self._getOriginAndNormal(modelNo)

        #clipTest = [[(m,nn,oo) for nn, oo in zip(n,o)] for n,o,m in zip(normals,origins,models)]
        #print(normals, origins, clipTest,clippedModels, sep='\n\n')

        # There are several heart models, and for each model, there are several slices. Therefore the outer list
        # iterates through the models whilst the inner iterates through the slices generated from each model
        #clippedModels = [[m.clip(normal=nn,origin=oo) for nn, oo in zip(n,o)] for n,o,m in zip(normals,origins,models)]
        clippedModels = {k: [m.clip(normal=nn, origin=oo) for nn, oo in zip(
            n, o)] for n, o, m, k in zip(normals, origins, models, modelNo)}
        return clippedModels

    #### DISPLAY MESHES #####

    def modelView(self, modelNo=(0,)):
        '''
            Display the meshes of the models specified by modelNo
        '''

        p = pv.Plotter(shape=(len(modelNo), 1))
        models = self._readMesh(self._genModelPaths(modelNo))
        for i, model in enumerate(models):
            p.subplot(i, 0)
            p.add_mesh(model)

        p.show()
        return

    def sliceView(self, modelNo=(0,), lim=3):
        '''
            Display the slices of the models specified by modelNo. For each model,
            a maximum of 'lim' slices are shown
        '''
        # lim decides max num of slices plotted per model
        p = pv.Plotter(shape=(len(modelNo), lim))
        # Generate landmarks
        self._sphereMaker(modelNo)

        for i, model in enumerate(modelNo):
            counter = 0
            for j, (slyce, centers) in enumerate(zip(self.slices[model],self.landmarks[model])):
                if lim == counter:
                    break
                p.subplot(i, j)
                p.add_mesh(slyce)
                # Plot landmarks
                for center in centers:
                    p.add_mesh(center,color='red')
                counter += 1

        p.show()
        return

    def clipView(self, modelNo=(0,), lim=2):
        '''
            Display the clips of the models specified by modelNo. For each model,
            a maximum of 'lim' clips are shown
        '''
        p = pv.Plotter(shape=(len(modelNo), lim))
        # Generate landmarks
        self._sphereMaker(modelNo)

        clips = self._clipMaker(modelNo)
        for i, key in enumerate(modelNo):
            sampledClips = sample(list(zip(clips[key],self.landmarks[key])), lim)
            for j, (clip,centers) in enumerate(sampledClips):
                p.subplot(i, j)
                p.add_mesh(clip)
                for center in centers:
                    p.add_mesh(center, color='red')


        p.show()
        return

    def verificationView(self, modelNum=(0,)):
        '''
            Display the slices of the models specified by modelNo. This view is defined
            in order to assess the accuracy of the views generated
        '''
        assert len(modelNum) == 1
        lim = 3
        groups = [
            (2, np.s_[:])
        ]
        p = pv.Plotter(shape=(3, lim), groups=groups)
        self._sphereMaker(modelNum)
        # Get Clips
        clips = self._clipMaker(modelNum)
        # Sampled value indices matter because slices must correspond to clips later in the code
        enumSample = sample(list(enumerate(clips[modelNum[0]])), lim)
        pos, sampledClips = [], []
        for idx, val in enumSample:
            pos.append(idx)
            sampledClips.append(val)

        for i, (clip, centers) in enumerate(zip(sampledClips,self.landmarks[modelNum[0]])):
            p.subplot(0, i)
            p.add_mesh(clip)
            # plot landmarks
            for center in centers:
                p.add_mesh(center, color='red')

        # Get Slices
        for i in pos:
            p.subplot(1, i)
            p.add_mesh(self.slices[modelNum[0]][i])
            # plot landmarks
            for centers in self.landmarks[modelNum[0]]:
                for center in centers:
                    p.add_mesh(center, color='red')

            # What does slices return? You must index that and get the slices at pos

        # Get model
        p.subplot(2, 0)
        model = self._readMesh(self._genModelPaths(modelNum))[0]
        labels = (1,2,6,8,10)
        for l in labels:
            p.add_mesh(model.threshold((l, l)))
            

        p.show()
        return


    def thresholdView(self, labels: tuple, modelNum=(0,)):
        p = pv.Plotter()

        model = self._readMesh(self._genModelPaths(modelNum))[0]
        for l in labels:
            p.add_mesh(model.threshold((l, l)))

        p.show()
        return