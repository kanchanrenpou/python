from curses.panel import new_panel
import cv2
import numpy as np
W = 640
H = 480

device = 0

def sq_dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

if __name__ == '__main__':
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    if not cap.isOpened():
        print('Not Opened Video Camera')
        exit()
    
    count = 0
    c_l, c_h = 70, 190
    searching = True
    while searching:
        ret, img = cap.read()
        
        if not ret:
            print('Video Capture Error')
            continue
        try:
            img = cv2.autorotate(img, device)
        except:
            pass
            
        H, W = img.shape[:2]
        
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

        # cv2.imshow('bin', img_otsu)
        # continue

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = np.array((img_hsv[:, :, 0] >= c_l) * (img_hsv[:, :, 0] <= c_h), dtype=np.uint8)
        
        # img_blue = img * cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # cv2.imshow('blue', img_blue)
        # continue
        
        _, img_b2 = cv2.threshold(img_gray * mask, 0, 225, cv2.THRESH_OTSU)
        try:     # for iPhone
            contours, _ = cv2.findContours(img_b2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:  # for PC
            _, contours, _ = cv2.findContours(img_b2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_plot = img.copy()

        # 面積最大のものを選択
        if len(contours) == 0:
            continue
        card_cnt = max(contours, key=cv2.contourArea)

        # 画像に輪郭を描画
        line_color = (255, 0, 0)
        thickness = 2
        cv2.drawContours(img_plot, [card_cnt], -1, line_color, thickness)
        
        epsilon = 0.1*cv2.arcLength(card_cnt, True)
        approx = cv2.approxPolyDP(card_cnt, epsilon, True)
        
        img_plot = np.zeros((480, 1920), dtype=np.uint8)
        img_plot[:, :640] = img_hsv[:, :, 0]
        img_plot[:, 640:1280] = img_hsv[:, :, 1]
        img_plot[:, 1280:] = img_hsv[:, :, 2]
        
        # img_plot = img_hsv.copy()
        # img_plot[:, :, 0] += 100
        # img_plot = cv2.cvtColor(img_plot, cv2.COLOR_HSV2RGB)[:,:,0]
        # img_plot = (img_gray * mask).copy()
        
        
        if len(approx) == 4:
            cv2.drawContours(img_plot, [approx], -1, line_color, thickness)
            count += 1
        else:
            count = 0
        
        k = cv2.waitKey(1) & 0xFF
        searching = (count <= 30) and (k != ord("q"))
        
        cv2.imshow('Final Result', img_plot[:, ::-1])
        try:
            key = cv2.waitKey(10)
        except:
            continue
        # 
    print(approx)
    
    img_out = img.copy()
    for point in approx:
        x = point[0][0]
        y = point[0][1]
        
        img_out = cv2.circle(img_out, (x,y), 10, (255,0,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_out = cv2.putText(img_out, "{}, {}".format(x, y), (x, y), font, 1.0, (255,255,255),2,cv2.LINE_AA)
        
    cv2.imshow("Output", img_out)
    key = cv2.waitKey(0)
    
    points = np.array([[p[0][0], p[0][1]] for p in approx], dtype=np.float32)
    
    if sq_dist(points[0], points[1]) >= sq_dist(points[1], points[2]):
        dst_p = [[840, 10], [10, 10], [10, 565], [840, 565]]
    else:
        dst_p = [[10, 10], [10, 565], [840, 565], [840, 10]]
        
    dst_p = np.array(dst_p, dtype=np.float32)
        
    mat = cv2.getPerspectiveTransform(points, dst_p)
    perspective_img = cv2.warpPerspective(img, mat, (850, 575))
    
    cv2.imshow("Calibrated", perspective_img)
    key = cv2.waitKey(0)
    
    