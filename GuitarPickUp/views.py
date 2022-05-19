import json
from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from django.http import HttpResponseRedirect
import cv2
from django.contrib import messages
from django.http.response import JsonResponse
from django.shortcuts import get_object_or_404, render
import threading
import mediapipe as mp
import time
from django.db import models
from django.shortcuts import redirect, render
from django.views.generic.list import ListView
from django.views.generic.detail import DetailView
from django.views.generic.edit import CreateView,UpdateView,DeleteView,FormView
from django.urls import reverse_lazy
from django.contrib import messages
from .tuner_hps import *
from django.contrib.auth.views import LoginView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login,authenticate
from django.shortcuts import render, redirect
from .forms import LoginForm, RegisterSerializer, UserSerializer
from .forms import CreateUserForm
from rest_framework import generics, permissions
from rest_framework.response import Response
from knox.models import AuthToken
from rest_framework.authtoken.serializers import AuthTokenSerializer
from knox.views import LoginView as KnoxLoginView
#imports from adel
import joblib
import numpy as np
import threading
from django.http import JsonResponse

from django.conf import settings 

def home(request):
    if (request.method == 'POST'):
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password1')
        user = User.objects.create_user(username, email, password)
        user.save()
    
    exercises = Excercise.objects.values('id','title','positions')
    feedback = None
    if(request.user.id != None):
        feedback = Feedback.objects.filter(user_id = request.user.id).values('id','feedback')#user_id = request.user.id
    
    #print(feedback.values())
    #print(exercises)
    return render(request, 'base/home.html',{'exercises':exercises,'feedbacks':feedback})

class CustomLoginView(LoginView):
    template_name = 'base/login.html'
    fields = '__all__'
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy('home')


def loginPage(request):
	if request.user.is_authenticated:
		return redirect('home')
	else:
		if request.method == 'POST':
			username = request.POST.get('username')
			password =request.POST.get('password')

			user = authenticate(request, username=username, password=password)

			if user is not None:
				login(request, user)
				return redirect('home')
			else:
				messages.info(request, 'Username OR password is incorrect')

		context = {}
		return render(request, 'base/login.html', context)


class RegisterAPI(generics.GenericAPIView):
    serializer_class = RegisterSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response({
        "user": UserSerializer(user, context=self.get_serializer_context()).data,
        "token": AuthToken.objects.create(user)[1]
        })

class LoginAPI(KnoxLoginView):
    permission_classes = (permissions.AllowAny,)

    def post(self, request, format=None):
        serializer = AuthTokenSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        login(request, user)
        return super(LoginAPI, self).post(request, format=None)

# def registerPage(request):
# 	if request.user.is_authenticated:
# 		return redirect('home')
# 	else:
# 		if request.method == 'POST':
#             form = CreateUserForm(request.POST)
#             if form.is_valid():
#                 form.save()
#                 user = form.cleaned_data.get('username')
#                 messages.success(request, 'Account was created for ' + user)
#                 return redirect('login')
			

# 		context = {'form':form}
# 		return render(request, 'base/home.html', context)

def coursePage(request):
    #t1 = threading.Thread(target=sound)
    #t1.start()
    #print(request.user.id)
    #feedback = Feedback(feedback = "test" , report = "test",user_id = request.user)
    #feedback.save()
    
    
    return render(request, 'base/try_excercise.html')

def feedbackpage(request):
    return render(request, 'base/feedback.html')

def pyscripttest(request):
    return render(request, 'base/testpyscript.html')

def tuner(request):
    return render(request, 'base/testTuner.html')

@gzip.gzip_page
def mediapipePage(request):
    cam = VideoCamera()
    t1 = threading.Thread(target=sound)
    t1.start()
    return StreamingHttpResponse(gen(cam, request), content_type="multipart/x-mixed-replace;boundary=frame")


index_model = joblib.load("GuitarPickUp/models/classifier_index.pkl")
middle_model = joblib.load("GuitarPickUp/models/classifier_middle.pkl")
ring_model = joblib.load("GuitarPickUp/models/classifier_ring.pkl")
pinky_model = joblib.load("GuitarPickUp/models/classifier_pinky.pkl")

def validate_hands(request):
    left_hand = request.POST.get('left_hand', None)
    right_hand = request.POST.get('right_hand', None)
    data = {}
    if (right_hand):
        right_decoded = json.loads(right_hand)
        index_left_coor = np.array([right_decoded[5]['x'],right_decoded[5]['y'],right_decoded[5]['z'],
            right_decoded[6]['x'],right_decoded[6]['y'],right_decoded[6]['z'],
            right_decoded[7]['x'],right_decoded[7]['y'],right_decoded[7]['z'],
            right_decoded[8]['x'],right_decoded[8]['y'],right_decoded[8]['z']])
        index_left_coor = index_left_coor.reshape(1,12)
        middle_left_coor = np.array([right_decoded[9]['x'],right_decoded[9]['y'],right_decoded[9]['z'],
            right_decoded[10]['x'],right_decoded[10]['y'],right_decoded[10]['z'],
            right_decoded[11]['x'],right_decoded[11]['y'],right_decoded[11]['z'],
            right_decoded[12]['x'],right_decoded[12]['y'],right_decoded[12]['z']])
        middle_left_coor = middle_left_coor.reshape(1,12)

        ring_left_coor = np.array([right_decoded[13]['x'],right_decoded[13]['y'],right_decoded[13]['z'],
            right_decoded[14]['x'],right_decoded[14]['y'],right_decoded[14]['z'],
            right_decoded[15]['x'],right_decoded[15]['y'],right_decoded[15]['z'],
            right_decoded[16]['x'],right_decoded[16]['y'],right_decoded[16]['z']])
        ring_left_coor = ring_left_coor.reshape(1,12)
                    
        pinky_left_coor = np.array([right_decoded[17]['x'],right_decoded[17]['y'],right_decoded[17]['z'],
            right_decoded[18]['x'],right_decoded[18]['y'],right_decoded[18]['z'],
            right_decoded[19]['x'],right_decoded[19]['y'],right_decoded[19]['z'],
            right_decoded[20]['x'],right_decoded[20]['y'],right_decoded[20]['z']])
        pinky_left_coor = pinky_left_coor.reshape(1,12)
                    
                    
        #print(index_left_coor)
        index_prediction = index_model.predict(index_left_coor)[0]
        middle_prediction = middle_model.predict(middle_left_coor)[0]
        ring_prediction = ring_model.predict(ring_left_coor)[0]
        pinky_prediction = pinky_model.predict(pinky_left_coor)[0]
        info = getInfo()
        data = {
            'index': index_prediction,
            'middle': middle_prediction,
            'ring': ring_prediction,
            'pinky': pinky_prediction,
            'note': info,
        }
    return JsonResponse(data)


def record_feedback(request):
    '''
    index_class = request.GET.get('index_class')
    middle_class = request.GET.get('middle_class')
    ring_class = request.GET.get('middle_class')
    pinky_class = request.GET.get('pinky_class')
    note_played = request.GET.get('note_played')
    index_bool = index_class == 'correct'
    middle_bool = middle_class == 'correct'
    ring_bool = ring_class == 'correct'
    pinky_bool = pinky_class == 'correct'
    '''
    #feedback = Feedback(feedback = "test" , report = "test",user_id = request.user)
    #feedback.save()
    '''
    last_feedback_id = Feedback.objects.latest('id')
    feedback_details = Feedback_details(feedback_id = last_feedback_id,index_class = index_bool,
                            middle_class = middle_bool,
                            ring_class = ring_bool,
                             pinky_class = pinky_bool,
                             note_played = note_played)
    feedback_details.save()
    '''
    feedback_root = settings.FEEDBACK_URL
    last_feedback_id = str(int(str(Feedback.objects.latest('id'))) + 1)
    full_path = feedback_root + last_feedback_id
    feedback = Feedback(feedback = full_path , report = "test",user_id = request.user)
    feedback.save()
    
    #last_feedback_id = str(Feedback.objects.latest('id'))

    positions = request.POST.get('positions')
    #replace sharps with just a hash
    positions = positions.replace('\u266f', '#')
    with open(f'{feedback_root}{last_feedback_id}','w') as f:
        json.dump(positions,f)
    messages.success(request, "feedback saved!")
    
    return JsonResponse({'ok':1})


#to capture video class
class VideoCamera(object):
    index_model = joblib.load("GuitarPickUp/models/classifier_index.pkl")
    middle_model = joblib.load("GuitarPickUp/models/classifier_middle.pkl")
    ring_model = joblib.load("GuitarPickUp/models/classifier_ring.pkl")
    pinky_model = joblib.load("GuitarPickUp/models/classifier_pinky.pkl")
    def __init__(self):
        self.detector = handDetector()
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frames) = self.video.read()
        self.frame = self.detector.find_hands(self.frames)
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        
        while True:
            (self.grabbed, self.frames) = self.video.read()
            self.frame = self.detector.find_hands(self.frames)
            hands_dict = self.detector.find_position2(self.frame)
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
                
                
                print(index_left_coor)
                index_prediction = self.index_model.predict(index_left_coor)[0]
                middle_prediction = self.middle_model.predict(middle_left_coor)[0]
                ring_prediction = self.ring_model.predict(ring_left_coor)[0]
                pinky_prediction = self.pinky_model.predict(pinky_left_coor)[0]
                
                cv2.putText(self.frame,f"index {index_prediction}",(10,70), cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
                cv2.putText(self.frame,f"middle {middle_prediction}",(10,110), cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
                cv2.putText(self.frame,f"ring {ring_prediction}",(10,150), cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
                cv2.putText(self.frame,f"pinky {pinky_prediction}",(10,190), cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
            #cv2.imwrite(f'{dirName}/frame_{frameNr}.jpg',img)
            #frameNr = frameNr+1
            



def sound(request):
    #with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
        #while True:
            #time.sleep(0.5)
    
    amp = request.POST.get('amp')
    
    amp_dict = json.loads(amp)
    #indata = np.array([ np.float32((s>>2)/(32768.0)) for s in amp_dict.values()])
    indata = np.array([s for s in amp_dict.values()])
    indata = indata.reshape(-1,1)

    note = callback(indata,None,200,False)
    print(note)
    note_response = {'indata':note}
    return JsonResponse(note_response)

    
    
    


def gen(camera, rq):
    while True:
        if (rq.path != "/mediapipePage/"):
            print("hi")
            break
        frame = camera.get_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

class handDetector():
    def __init__(self,mode = False,max_hands = 2,detection_confidence = 0.5,trackConfidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        
        self.hands = self.mpHands.Hands(self.mode,self.max_hands,1,self.detection_confidence,self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def mirror_this(image_file, gray_scale=False, with_plot=False):
        image_mirror = np.fliplr(image_file)
        return image_mirror.astype(np.uint8).copy() 
    
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
    
def record(request):
    if request.method == "POST":
        
        video_file = request.FILES.get("excercise_video")
        record = StudentVideo.objects.create(video_record=video_file)
        record.save()
        
        messages.success(request, "Video successfully added!")
        
        return JsonResponse(
            {
                "success": True,
            }
        )
        
    return render(request, "base/home.html")