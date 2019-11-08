import SimpleITK as sitk
import numpy as np
import pickle
import sparse

class image:
    '''
    This class is originally defined to load CCF atlas (value of a pixel is the ID of a brain structure)
    '''
    def __init__(self, file=None, pickle_file=None):
        assert ((file is not None) | (pickle_file is not None)), "Error in initializing image object."
        if pickle_file is None:
            self.file = file
            self.image = sitk.ReadImage(file)
        else:
            '''
            To be deprecated: Save large image files as numpy array takes to much space
            '''
            self.file = "Given image"
            array, spacing = pickle.load(open(pickle_file, 'rb')) # Note that pickle does not support SimpleITK image class
            if type(array) == sparse.coo.core.COO:
                array = array.todense()
            array = np.swapaxes(array, 0, 2)
            self.image = sitk.GetImageFromArray(array)
            self.image.SetSpacing(spacing)
        self.array = sitk.GetArrayViewFromImage(self.image)
        self.array = np.swapaxes(self.array, 0, 2) # Swap XZ axis to meet the convention of V3D
        # self.values = np.unique(self.array).tolist()
        self.axes = ["x", "y", "z"]
        self.size = dict(zip(self.axes, self.image.GetSize()))
        self.space = dict(zip(self.axes, self.image.GetSpacing()))
        self.micron_size = dict(zip(self.axes,
                                    [self.size["x"] * self.space["x"],
                                     self.size["y"] * self.space["y"],
                                     self.size["z"] * self.space["z"],
                                     ]
                                    ))
    
    
    def writeNRRD(input_M, IDlist, path, flipF = True):
        '''
        Write interested region in a 3D array to nrrd file, with interested region to be 1
        :param input_M: an 3D array, the value at each point is the ID of specific region
        :param IDlist: a list containing interested value
        :param path: the destination to store the output
        :param range_y:
        :param flipF: whether illustrating half part of the result or to illustrate the full result.
                        It does not influce the saving result.
        :return: No
        '''
        assert type(IDlist) ==list, "Please using list to input the interested regionID"
        folder = os.path.exists(path)
        if not folder:         
            os.makedirs(path)  
        data = np.zeros(input_M.shape)
        for iterID in IDlist:
            data[input_M==iterID]=1
        filename = str(path)+ '/'
        for iterID in IDlist:
            filename = filename + str(iterID)
            filename = filename + '_'
        filename = filename +'.nrrd'
        itkimage = sitk.GetImageFromArray(data, isVector=False)
        sitk.WriteImage(itkimage, filename, True) 
        co_1,co_2,co_3 = np.where(data ==1)
        if flipF:
            co_1 = co_1[co_3<=nmt.annotation.size['z']//2]
            co_2 = co_2[co_3<=nmt.annotation.size['z']//2]
            co_3 = co_3[co_3<=nmt.annotation.size['z']//2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Plot defining corner points
        ax.plot(co_1, co_2,co_3, "y.")
        # Make axis label
        for i in ["x", "y", "z"]:
            eval("ax.set_{:s}label('{:s}')".format(i, i))
        plt.show()



