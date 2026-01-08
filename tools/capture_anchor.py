import os
import cv2
import numpy as np
import mss

OUT_PATH = "assets/templates/anchor.png"

def getAnchor():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with mss.mss() as sct:
        mon = sct.monitors[1]
        img = np.array(sct.grab(mon))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    cv2.imshow("Select ANCHOR ROI (press ENTER to confirm, ESC to cancel)", frame)
    roi = cv2.selectROI("Select ANCHOR ROI (press ENTER to confirm, ESC to cancel)", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        print("Canceled.")
        return

    anchor = frame[y:y+h, x:x+w].copy()
    cv2.imwrite(OUT_PATH, anchor)
    print("Saved anchor to:", OUT_PATH)
    
def getKP():
    anchor = cv2.imread("assets/templates/anchor.png", cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=6000)
    kpA, desA = orb.detectAndCompute(anchor, None)
    print(len(kpA))
    if desA is None or len(kpA) < 20:
        raise RuntimeError("Anchor has too few features. Choose a more textured ROI.")
        

if __name__ == "__main__":
    getKP()
