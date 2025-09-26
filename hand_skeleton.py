import cv2, numpy as np

def skin_mask(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], np.uint8)
    upper = np.array([255, 180, 135], np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return mask

def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea)

def unit(v):
    n = np.linalg.norm(v)
    return v/n if n>1e-6 else v

def line_endpoints(mask, center, normal, max_len=200):
    h,w = mask.shape
    n = unit(normal); c = np.array(center, float)

    def march(sign):
        p = c.copy()
        for _ in range(max_len):
            p += sign*n
            x,y = int(round(p[0])), int(round(p[1]))
            if x<0 or y<0 or x>=w or y>=h: break
            if mask[y,x]==0:
                p -= sign*n; break
        return np.array([int(p[0]), int(p[1])])
    return march(-1), march(+1)

def draw_circle(img, p, color, r=3):
    cv2.circle(img, tuple(int(x) for x in p), r, color, -1, lineType=cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera"); return

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        H,W = frame.shape[:2]

        mask = skin_mask(frame)
        cnt = largest_contour(mask)
        vis = frame.copy()

        if cnt is not None and cv2.contourArea(cnt) > 1000:
            # จุด centroid = ศูนย์กลางฝ่ามือ
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                palm = np.array([cx,cy])
                draw_circle(vis, palm, (0,0,255), 5)

                # ทดลอง 5 เส้นแนวตั้งฉาก (แทน MCP, PIP, DIP, TIP, TIP+)
                for ratio in [0.3,0.55,0.8,1.0]:
                    # สมมติทิศทางนิ้ว = straight up (0,-1)
                    v = np.array([0,-1])
                    base = palm + v*int(H*ratio*0.5)
                    perp = np.array([-v[1], v[0]])
                    L,R = line_endpoints(mask, base, perp, 300)
                    center = ((L+R)//2).astype(int)

                    cv2.line(vis, tuple(L), tuple(R), (255,255,255), 2) # เส้นขาว
                    draw_circle(vis, L, (255,150,0), 3)  # จุดฟ้า
                    draw_circle(vis, R, (255,150,0), 3)
                    draw_circle(vis, center, (0,0,255), 4)  # จุดแดง

        cv2.imshow("Hand skeleton idea", vis)
        k = cv2.waitKey(1) & 0xFF
        if k==ord('q'): break
        if k==ord('s'):
            cv2.imwrite("output_samples/snapshot.png", vis)
            print("💾 Saved snapshot.png")

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
