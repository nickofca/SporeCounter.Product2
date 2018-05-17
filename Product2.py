# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:05:25 2018

Outer circle volume = 26.44609669	

Hyperparameters:
    
    SMin- Size Disclusion Minimum pixel size
        Best:95
    SMax- Size Disclusion Maximum pixel size
        Best:880
    InnerScalar- Scalar from scope radius to reading circle radius
    
Consider a debris discluder from overall volume 

@author: Anderson Lab
"""



import cv2
import numpy as np
from scipy import ndimage
import imageio
import PIL


    
def main():
    global innerScalar
    SMin = 95
    SMax = 880
    verbose = True
    innerScalar = 0.65
    innerVol = 26.44609669*innerScalar**2
    #File location
    imloc=r'C:\Users\Anderson Lab\Desktop\Nick Z\SporeCounter.Product1\Pictures\Nosema3.jpg'
    #Read image and create color copy
    img = cv2.imread(imloc,0)
    cimg = cv2.imread(imloc,1)
    #Find inner focal circle
    circleImage, medianCircle = circleFinder(img,cimg)
    #Make mask
    zeros = np.zeros(img.shape)
        #Scale inside circle
    medianCircle[0,2]= np.around(np.multiply(medianCircle[0,2],innerScalar))
    mask = cv2.circle(zeros.copy(), (medianCircle[0,0],medianCircle[0,1]),medianCircle[0,2], (255,255,255),-1, 8,0).astype(np.uint8)
    circle = cv2.circle(zeros.copy(), (medianCircle[0,0],medianCircle[0,1]),medianCircle[0,2], (255,255,255), 5).astype(np.uint8)
    #Implement mask 
    img = np.multiply(np.divide(mask,255).astype(np.uint8),img)
    #Testing with blur
    blur = cv2.bilateralFilter(img,9,80,80)
    #Mean Thresholding
    thr = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,1)
    thr = np.multiply((1-np.divide(circle,255)),thr).astype(np.uint8)
    #Open structure
    kernel = np.ones((4,4),np.uint8)
    closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel) 
    #Size disclusion
    labeledfiltered = (sizeDisc(closing,SMin,SMax)+0).astype(np.uint8)
    #Count
    count = ndimage.label(labeledfiltered)[1]
        #visualize overlay of found over original
        #Report number found
    print("Spores found:")
    print(count)
    print()
    print("Using a 11.1735 nL volume for the inner circle")
    print()
    print("Spore concentration:")
    print(f"{count/innerVol} [spores/nL]")
    #cpictify(overlayer(cimg,edges_binary))
    if verbose:
        kernel = np.ones((2,2),np.uint8)
        overlay = overlayer(cimg,cv2.erode(labeledfiltered,kernel, iterations =1))
        pictify(closing)
        cpictify(overlay)
    #pictify(thr)


#diagnostic function to turn array to picture window
def cpictify(array):
    if array.max()==1:
        array=(array*255).astype(np.uint8)
    
    im_temp2=PIL.Image.fromarray(array,'RGB')
    im_temp2.show()
def pictify(array):
    if array.max()==1:
        array=(array*255).astype(np.uint8)
    
    im_temp2=PIL.Image.fromarray(array,'L')
    im_temp2.show()
#function to save array as jpg
def savepic(array,name):
    imageio.imwrite(name+".jpg",array)
#overlays binary onto color image
def overlayer(cimg2,overlayIO):
    cimg2[:,:,0] = np.where(overlayIO!=0, (overlayIO*255).astype(np.uint8), cimg2[:,:,0])
    cimg2[:,:,1] = np.where(overlayIO!=0, np.divide(cimg2[:,:,1],2), cimg2[:,:,1])
    cimg2[:,:,1] = np.where(overlayIO!=0, np.divide(cimg2[:,:,2],2), cimg2[:,:,2])
    return(cimg2)
 def supervise(array,cimg):
    #labels non-connected bodies 
    IODict = {}
    overlaidDict = {}
    for index in range(1,100):
        overlaid, IO = seperate(array,index,cimg)
        count = float(trainPictify(overlaid))
        file = os.path.splitext(os.path.basename(imloc))[0]
        #save image and array in folder
        overlaidPath = f"colorTrain\\{file}\\{index}"
        IOPath = f"IOTrain\\{file}\\{index}"
        zDirCreate(f"colorTrain\\{file}")
        zDirCreate(f"IOTrain\\{file}")
        savepic(overlaid,overlaidPath)
        savepic(IO,IOPath)
        #update dictionary
        IODict.update({os.getcwd()+"\\"+IOPath+".jpg":count})
        overlaidDict.update({os.getcwd()+"\\"+overlaidPath+".jpg":count})
    #Save json
    with open(f'IOTrain\\{file}\\data.json', 'w') as fp:
        json.dump(IODict, fp, sort_keys=True, indent=4)
    with open(f'colorTrain\\{file}\\data.json', 'w') as fp:
        json.dump(overlaidDict, fp, sort_keys=True, indent=4)
    
def seperate(labeledfiltered,index,cimg):
    #incorporate parallel computing https://blog.dominodatalab.com/simple-parallelization/
    nlab = ndimage.label(labeledfiltered)[1]
    labs = ndimage.label(labeledfiltered)[0]
    #Center of mass approx
    centers = ndimage.center_of_mass(labeledfiltered,labs,list(np.arange(1,nlab+1)))
    #Create array of just object of interest
    label = labs.copy()
    label[label!=index] = 0
    label[label==index] = 1
    over = overlayer(cimg,label)
    overOut=ZcolShaper(centers,index,over)
    IO=Zshaper(centers,index,label)
    #parallel computing
    #num_cores = multiprocessing.cpu_count()
    #payload = Parallel(n_jobs=2, verbose=10, timeout= 10)(delayed(gather)(index=index, labs=labs, centers=centers) for index in np.arange(1,nlab+1))
    #It took 124.17280721664429 seconds/2.0695467869440716 minutes to compute
    return((overOut,IO))
    
#Run object through neural network
def trainBuild():
    #trainset(xdim,ydim,color,
    trainIO = np.zeros((200,200,1,nlab))
    trainColor = np.zeros((200,200,3,nlab))
    starttime = time.time()
    for i in range(1,nlab+1):
        out = seperate(labeledfiltered,index = i,cimg = cimg)
        trainIO[:,:,:,i-1]= np.atleast_3d(out[1])
        trainColor[:,:,:,i-1]= out[0]
        print(f"iteration: {i}")
    elapsedtime = time.time() - starttime
    print(f"It took {elapsedtime} seconds/{elapsedtime/60} minutes to compute")
    return((trainColor,trainIO))
    
def ZcolShaper(centers,index,label):
    out = np.zeros((200,200,label.shape[2]))
    center = centers[index-1]
    #Copy from labeled using copyArray = label[midpointx-size/2:midpointx+size/2,midpointy-size/2:midpointy+size/2]
    xmin = int(max(center[0]-100,0))
    xmax = int(min(center[0]+99,label.shape[0]))
    ymin = int(max(center[1]-100,0))
    ymax = int(min(center[1]+99,label.shape[1]))
    window = label[xmin:xmax,ymin:ymax,:]
    xwin = window.shape[0]/2
    ywin = window.shape[1]/2
    out[floor(100-xwin):floor(100+xwin),floor(100-ywin):floor(100+ywin),:] = window
    return(out.astype(np.uint8))
    
def Zshaper(centers,index,label):
    out = np.zeros((200,200))
    center = centers[index-1]
    #Copy from labeled using copyArray = label[midpointx-size/2:midpointx+size/2,midpointy-size/2:midpointy+size/2]
    xmin = int(max(center[0]-100,0))
    xmax = int(min(center[0]+99,label.shape[0]))
    ymin = int(max(center[1]-100,0))
    ymax = int(min(center[1]+99,label.shape[1]))
    window = label[xmin:xmax,ymin:ymax]
    xwin = window.shape[0]/2
    ywin = window.shape[1]/2
    out[floor(100-xwin):floor(100+xwin),floor(100-ywin):floor(100+ywin)] = window
    return(out.astype(np.uint8))
     
def circleFinder(img,cimg):
    img = cv2.resize(img,(0,0),fx=.2,fy=.2)
    img = cv2.medianBlur(img,5)
    __,thr = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,1,param1=500,param2=50,minRadius=100,maxRadius=min(img.shape))
    circles = np.uint16(np.around(circles))
    medianMid = np.uint16(np.around(np.mean(circles,axis=1)))
    #rescale
    medianMid = np.multiply(medianMid,5)
    i = medianMid
    cv2.circle(cimg,(i[:,0],i[:,1]),i[:,2],(0,255,0),5)
    cv2.circle(cimg,(i[:,0],i[:,1]),np.around(i[:,2]*innerScalar),(255,0,255),5)
    # draw the center of the circle
    cv2.circle(cimg,(i[:,0],i[:,1]),2,(0,0,255),6)
    return((cimg,medianMid))
       
def sizeDisc(array,SMin,SMax):
        #labels non-connected bodies uniquely (array)
    labeled = ndimage.label(array)
        #counts the size of each object (list)
    count = np.bincount(labeled[0].flatten())
        #size filter 
    countfiltered = np.logical_and(count>SMin,count<SMax)
        #list of approved object indeces
    approvedobjects = np.where(countfiltered)
        #array of only approved objects
    labeledfiltered = np.isin(labeled[0],approvedobjects)
    return(labeledfiltered)
    
if __name__ == "__main__":
    main()