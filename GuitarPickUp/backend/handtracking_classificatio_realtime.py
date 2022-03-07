

import cv2
import mediapipe as mp
import time
import csv
import joblib
import numpy as np

class handDetector():
    def __init__(self,mode = False,max_hands = 1,detection_confidence = 0.5,trackConfidence = 0.5):
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
            #print(self.results.multi_handedness)
            try:
                myHand = self.results
                # index,score,label
                handReport = myHand.multi_handedness
                #print(type(handReport[0]))
                myHandLandmarks = myHand.multi_hand_landmarks[handNo]
                print(myHandLandmarks)
                for id,lm in enumerate(myHand.landmark):
                    #print(id,lm)
                    
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w) , int(lm.y*h)
                    #print(id,cx,cy)
                    lmList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
            except Exception as exp:
                pass
                
                
        return lmList
            
            #id is the landmark
            #4,8,12,16,20
            #if(id == 4):
                #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
            #    print(id,str(cx*lm.z),str(cy*lm.z))
    def find_position2(self,img,draw = False):
        lmList = []
        if(self.results.multi_hand_landmarks):
            #print(self.results.multi_handedness)
        
            myHand = self.results
            # index,score,label
            handReports = myHand.multi_handedness
        
            
            
            multi_handedness_list = []
            for handReport in handReports:
                ls = [{'index':value.index,'label':value.label,'confidence':value.score} for value in handReport.classification]
                #the dictionary has to be returned in an array
                multi_handedness_list.append(ls[0])
            #print(multi_handedness_list)
            hands_dictionaries = []
            for index,hand in enumerate(multi_handedness_list):
                #print(hand)
                myHandLandmarks = myHand.multi_hand_landmarks[index]
                #print(myHandLandmarks)
                positions  = {'hand_class':hand['label'],
                              'hand_confidence':hand['confidence'],
                              'pos':[],'screen_width':img.shape[0],'screen_height':img.shape[1]}
                for id,lm in enumerate(myHandLandmarks.landmark):
                    positions['pos'].append((id,lm.x,lm.y,lm.z))
                    
                    #print(id,lm,hand['label'])

                hands_dictionaries.append(positions)
                
            
                #print(hands_dictionaries)
                    
                    
            return hands_dictionaries
                
                
            
            #print(len(myHandLandmarks))
            
            #for handLandmark,handReport in myHandLandmarks,handReports:
                #print(handLandmark)
            '''
            for id,lm in enumerate(myHandLandmarks.landmark):
                print(id,lm)
                
                h,w,c = img.shape
                cx,cy = int(lm.x*w) , int(lm.y*h)
                #print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
            '''
            
                
                
        
    
    '''
    #print(id,lm)
    h,w,c = img.shape
    cx,cy = int(lm.x*w) , int(lm.y*h)
    #print(id,cx,cy)
    
    #id is the landmark
    #4,8,12,16,20
    if(id == 4):
        #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        print(id,str(cx*lm.z),str(cy*lm.z))
        
    
    if(id == 8):
        #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        #print(id,cx,cy)
    if(id == 12):
        #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        #print(id,cx,cy)
    if(id == 16):
        #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        #print(id,cx,cy)
    if(id == 20):
        #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        #print(id,cx,cy)
    '''
    
def format_string(s):
    #x = s[1]
    #y = s[2]
    #id= s[0]
    #return {'id':id,'x':x,'y':y}
    return str(s).replace(',','').replace('[','').replace(']','')
def mirror_this(image_file, gray_scale=False, with_plot=False):

    image_mirror = np.fliplr(image_file)
    return image_mirror.astype(np.uint8).copy() 

def main():
    pTime = 0
    cTime = 0
    detector = handDetector(max_hands=2)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #captureDevice = camera
    index_model = joblib.load("classifier_index.pkl")
    middle_model = joblib.load("classifier_middle.pkl")
    ring_model = joblib.load("classifier_ring.pkl")
    pinky_model = joblib.load("classifier_pinky.pkl")
    while True:
        success, img = cap.read()
        #overwrite drawn image draw is set true by default
        img = detector.find_hands(img)
        #img_mirror = mirror_this(img)
        hands_dict = detector.find_position2(img)
        #print(hands_dict)
        coordinates_left = []
        coordinates_right = []
        
        screen_width = None
        screen_height = None
        #if hands are not detected dictionary is null which triggers an error
        if(hands_dict != None):
            #store left and right hand into separate lists depending on the dictionary
            if(len(hands_dict) == 1):
                screen_width = hands_dict[0]['screen_width']
                screen_height = hands_dict[0]['screen_height']
                if(hands_dict[0]['hand_class'] == 'Right'):
                    coordinates_right = hands_dict[0]['pos']
                if(hands_dict[0]['hand_class'] == 'Left'):
                    coordinates_left = hands_dict[0]['pos']
            if(len(hands_dict) == 2):
                screen_width = hands_dict[0]['screen_width']
                screen_height = hands_dict[0]['screen_height']
                if(hands_dict[0]['hand_class'] == 'Left' and hands_dict[1]['hand_class'] == 'Right'):
                    coordinates_left = hands_dict[0]['pos']
                    coordinates_right = hands_dict[1]['pos']
                if(hands_dict[0]['hand_class'] == 'Right' and hands_dict[1]['hand_class'] == 'Left'):
                    coordinates_right = hands_dict[0]['pos']
                    coordinates_left = hands_dict[1]['pos']
    
        #print(coordinates_left)
        if(len(coordinates_right) != 0):
            index_left_coor = np.array([coordinates_right[5][1],coordinates_right[5][2],coordinates_right[5][3],
                                        coordinates_right[6][1],coordinates_right[6][2],coordinates_right[6][3],
                                        coordinates_right[7][1],coordinates_right[7][2],coordinates_right[7][3],
                                        coordinates_right[8][1],coordinates_right[8][2],coordinates_right[8][3]])
            index_left_coor = index_left_coor.reshape(1,12)
            middle_left_coor = np.array([coordinates_right[9][1],coordinates_right[9][2],coordinates_right[9][3],
                                        coordinates_right[10][1],coordinates_right[10][2],coordinates_right[10][3],
                                        coordinates_right[11][1],coordinates_right[11][2],coordinates_right[11][3],
                                        coordinates_right[12][1],coordinates_right[12][2],coordinates_right[12][3]])
            middle_left_coor = middle_left_coor.reshape(1,12)
            
            ring_left_coor = np.array([coordinates_right[13][1],coordinates_right[13][2],coordinates_right[13][3],
                                        coordinates_right[14][1],coordinates_right[14][2],coordinates_right[14][3],
                                        coordinates_right[15][1],coordinates_right[15][2],coordinates_right[15][3],
                                        coordinates_right[16][1],coordinates_right[16][2],coordinates_right[16][3]])
            ring_left_coor = ring_left_coor.reshape(1,12)
            
            pinky_left_coor = np.array([coordinates_right[17][1],coordinates_right[17][2],coordinates_right[17][3],
                                        coordinates_right[18][1],coordinates_right[18][2],coordinates_right[18][3],
                                        coordinates_right[19][1],coordinates_right[19][2],coordinates_right[19][3],
                                        coordinates_right[20][1],coordinates_right[20][2],coordinates_right[20][3]])
            pinky_left_coor = pinky_left_coor.reshape(1,12)
         
            index_prediction = index_model.predict(index_left_coor)[0]
            middle_prediction = middle_model.predict(middle_left_coor)[0]
            ring_prediction = ring_model.predict(ring_left_coor)[0]
            pinky_prediction = pinky_model.predict(pinky_left_coor)[0]
            
            cv2.putText(img,f"index {index_prediction}",(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.putText(img,f"middle {middle_prediction}",(10,110), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.putText(img,f"ring {ring_prediction}",(10,150), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.putText(img,f"pinky {pinky_prediction}",(10,190), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

                
        cTime = time.time()
        fps = 1/ (cTime - pTime)
        pTime = cTime

        #cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        
        
        cv2.imshow("Image",img)
        if(cv2.waitKey(1) == 27):
            cv2.destroyWindow("Image")
            break

    
if __name__ == "__main__":
    main()

    