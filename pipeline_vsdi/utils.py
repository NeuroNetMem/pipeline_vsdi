import cv2


def find_bounding_rectangle(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return (x, y), (x + w, y + h)
    else:
        return None
    

def crop_to_bounding_box(video,mask):
    bounds = find_bounding_rectangle(mask.astype('uint8'))
    video = video[bounds[0][1]:bounds[1][1],bounds[0][0]:bounds[1][0],:] 
    mask = mask[bounds[0][1]:bounds[1][1],bounds[0][0]:bounds[1][0]] 

    return video, mask

def resize(mask_list):
    max_h = np.max([x.shape[0] for x in mask_list])
    max_w = np.max([x.shape[1] for x in mask_list])
    
    resized = []
    for m in mask_list:
        resized_mask = cv2.resize(m, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
        resized.append(resized_mask)
    
    return resized
