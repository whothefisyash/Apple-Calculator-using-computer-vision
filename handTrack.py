import cv2 as cv
import mediapipe as mp

class handDetector: #constructor handDetector
    
    def __init__(self,mode=False,maxHands=2,detectionCon=0.75,trackCon=0.75):
        # mode=True (hands in static image)
        # mode=False(continuous video stream)
        # detectionCon: Confidence threshold for detecting a hand.
        # Higher values make detection more reliable but may miss some hands.
        # trackCon: Confidence threshold for tracking a hand after initial detection.
        
        # detectionCon inc->system more selective,might miss some hands in less clear fames(less false positives)
        # detectionCon dec->system detects hands more easily(inc false positives)
        # trackCon inc->slow responses-better tracking once hand is detected
        # trackCon dec->tracking fast but kess accurate hand position detection during motion


        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        self.mpHands=mp.solutions.hands #has pretrained models for detecting hands

        self.hands=self.mpHands.Hands(static_image_mode=self.mode ,
                                      max_num_hands=self.maxHands,
                                      min_detection_confidence=self.detectionCon,
                                      min_tracking_confidence=self.trackCon,
                                      )
        self.mpDraw=mp.solutions.drawing_utils

        self.tipIds=[4,8,12,16,20] #thumb,index,....,pinky

    
    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    
    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(
                        img, (cx, cy), 5, (255, 0, 255), cv.FILLED
                    )  # Smaller circle for less drawing overhead
        return self.lmList

    
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Reduce the resolution for faster processing
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()