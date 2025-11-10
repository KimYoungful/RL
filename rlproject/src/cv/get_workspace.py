from numpy import array



def get_workspace(img:array):
    # Get the height and width of the image
    height, width = img.shape[:2]

    # Define the workspace as a rectangle in the center of the image
    workspace = img[:-100,600:-500 ]
    # workspace =img[:,:] 

    return workspace