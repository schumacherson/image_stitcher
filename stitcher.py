import numpy as np
import cv2
import argparse
import glob
import sys

#the algorithm used to extract keypoints and features of the images. you can use: 'sift', 'orb' and 'brisk'
feature_extractor = 'sift'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="path to input directory of images to stitch")
ap.add_argument("-s", "--scale", type=float, default = 1, help="scale aplied to source images to shorten image processing time")
ap.add_argument("-e", "--ext", type=str, default=".jpg", help="extension of your images")
ap.add_argument("-t", "--threshold", type=int, default=50, help="number of minimum matches between images to relate them")
args = vars(ap.parse_args())

scale = args["scale"]
path = args["images"]
ext = args["ext"]
threshold = args["threshold"]

def main():
    print("Reading images...")

    images = [cv2.imread(file) for 
            file in sorted(glob.glob(path + r'/*'+ext))]

    #the size of the images is scaled, so the algorithm runs faster, as default, the scale factor is 1
    images = [cv2.resize(images[i], (int(images[i].shape[1]*scale), int(images[i].shape[0]*scale))) 
            for i in range(len(images))]

    if len(images) == 0:
        sys.exit("there is no images with the "+ext+" extension in the path you\'ve entered")

    print("Processing images...")
    graph, breadth_path, infoImg, matching = createGraph(images)

    homographies, each_path = getHomographies(breadth_path, infoImg, matching, graph)

    translate, finals, width, heigth = getRes(images, each_path, breadth_path, homographies)
    
    imagesCopy = images.copy()

    to_blend = []
    print("Warping and blending images...")

    #each image is transformed, moving them to their final position in the image mosaic
    for i in range(len(breadth_path)):
        img = imagesCopy[breadth_path[i]]
        
        final = translate.copy()
                
        final = np.dot(final, finals[breadth_path[i]])
        
        transformed = cv2.warpPerspective(img, final, (width, heigth))
        
        to_blend.append(transformed)
        blending = to_blend.copy()

    

    #the images are blended to form the final image mosaic
    for i in range(1, len(to_blend)):
        imgA = blending[i-1]
        imgB = blending[i]
            
        mask = get_mask(imgB)
        
        result = merge(imgA, imgB, mask)
            
        blending[i] = result
        
    cv2.imwrite(path + r'/result.'+ext, blending[-1])
    print("Done!")

# the final image resolution is found, by transforming the 4 edges of every images.
# having the min and max points we can find the rectangle which the image is located
# since there is a central image, there can be images located in areas with negative coordenates
# the translate matrix translates the rectangle containing the image to a positive coordinate system 
def getRes(images, each_path, breadth_path, homografias):
    xs = []
    ys = []
    finals = {}

    imagesCopy = images.copy()

    for i in range(len(breadth_path)):
        w = imagesCopy[breadth_path[i]].shape[1]
        h = imagesCopy[breadth_path[i]].shape[0]
        
        orderAc = each_path[breadth_path[i]].copy()
        orderAc.reverse()
        orderAc.append(breadth_path[i])
        
        final = np.identity(3)
        
        if len(orderAc) > 1:
            for j in range(len(orderAc) - 1):
                sup = homografias[(orderAc[j], orderAc[j+1])]
                final = np.dot(final, sup)
        
        finals[breadth_path[i]] = final
        
        p1 = [0, 0, 1]
        p2 = [w, 0, 1]
        p3 = [0, h, 1]
        p4 = [w, h, 1]
        p = [p1, p2, p3, p4]
        
        for pi in p:
            res = np.dot(final, pi)
            xs.append(res[0])
            ys.append(res[1])
        
    width = int(max(xs) - min(xs)) + 1
    heigth = int(max(ys) - min(ys)) + 1

    translate = np.array([
    [1, 0, -min(xs)],
    [0, 1, -min(ys)],
    [0, 0, 1]])

    return translate, finals, width, heigth

# the homographies that transform each image is found
def getHomographies(breadth_path, infoImg, matching, graph):
    used = []
    homographies = {} 
    each_path = {} 

    for i in range(len(breadth_path)):
        each_path[breadth_path[i]] = []

    for i in range(len(breadth_path)):
        for j in range(len(graph[breadth_path[i]])):
            if  graph[breadth_path[i]][j] not in used:
                
                infoA = infoImg[breadth_path[i]]
                infoB = infoImg[graph[breadth_path[i]][j]]

                (kpsA, featuresA) = infoA
                (kpsB, featuresB) = infoB

                matches = matching[(breadth_path[i], graph[breadth_path[i]][j])]

                H = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh = 4)
                
                homographies[(breadth_path[i], graph[breadth_path[i]][j])] = H #salva as homografias
                
                each_path[graph[breadth_path[i]][j]].append(breadth_path[i])
                
                for k in range(len(each_path[breadth_path[i]])):
                    each_path[graph[breadth_path[i]][j]].append(each_path[breadth_path[i]][k])
                
                used.append(graph[breadth_path[i]][j])
        
        used.append(breadth_path[i])

    return homographies, each_path

# creates a dictionary that represents the graph which describes the relation between images
def createGraph(images):
    graph = {}

    infoImg = {}

    matching = {}

    nbr_imgs = len(images)

    for i in range(nbr_imgs):
        graph[i] = [] 

    for i in range(0, nbr_imgs):
        for j in range(0, nbr_imgs):
            imgA = images[i]
            imgB = images[j]
            
            kpsA, featuresA = detectAndDescribe(imgA, method=feature_extractor)
            kpsB, featuresB = detectAndDescribe(imgB, method=feature_extractor)
            
            infoImg[i] = (kpsA, featuresA)
            infoImg[j] = (kpsB, featuresB)

            matches = matchKeypointsKNN(featuresA, featuresB, ratio=0.3, method=feature_extractor)
                
            if len(matches) > threshold and i!=j: #cria a aresta entre vértices caso cumpram a condição
                graph[i].append(j)
                matching[(i, j)] = matches
            
    choiceArray = [len(graph[i]) for i in range(len(images))]

    choice = choiceArray.index(max(choiceArray))

    breadth_path = breadth(graph, choice) #ordem de colagem das imagens
    
    return graph, breadth_path, infoImg, matching

# creates the object that can extract an image keypoints and features, and returns them
def detectAndDescribe (image, method=None):
        
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
        
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)

# creates the object that matches de keypoints between two images
def createMatcher(method, crossCheck):
    if method == 'sift':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    
    return bf

# match the keypoints between two images
def matchKeypointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    matches = []
    
    for m,n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches  

# realizes the breadth search algorithm in a graph
def breadth(grafo, inicio):
    visited = [False]*len(grafo)
    
    final = []
    
    queue = []
    
    queue.append(inicio)
    visited[inicio] = True
    
    while queue:
        inicio = queue.pop(0)
        final.append(inicio)
        
        for i in grafo[inicio]:
            if visited[i] == False:
                queue.append(i)
                visited[i] = True
                
    return final
    
# returns the homography that solve the system AX = B, where B is the keypoints coordinates 
# from the first image and A is the keypoints coordinates from the second image 
def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reprojThresh)

        return H
    else:
        return None
    
# gets the mask of an image
def get_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)[1]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask.astype(np.float32)
    mask = 1 - mask
    
    return mask

# blends two images based on their masks
def merge(imgA, imgB, mask):
    maskB = 1 - mask
    
    final = imgA * mask + imgB * maskB
        
    return final

if __name__ == "__main__":
    main()