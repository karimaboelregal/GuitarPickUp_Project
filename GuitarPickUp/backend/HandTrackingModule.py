import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,mode = False,max_hands = 2,detection_confidence = 0.5,trackConfidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        
        self.hands = self.mpHands.Hands(self.mode,self.max_hands,1,self.detection_confidence,self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        
    def find_hands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        
        if(self.results.multi_hand_landmarks):
            for handLms in self.results.multi_hand_landmarks:                
                if(draw):
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def find_position(self,img,handNo = 0,draw = False):
        lmList = []
        if(self.results.multi_hand_landmarks):
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id,lm in enumerate(myHand.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w) , int(lm.y*h)
                #print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
                
        return lmList
            
            #id is the landmark
            #4,8,12,16,20
            #if(id == 4):
                #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
            #    print(id,str(cx*lm.z),str(cy*lm.z))
    
    