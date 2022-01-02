from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from django.http import HttpResponseRedirect
import cv2
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

from django.contrib.auth.views import LoginView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login,authenticate
from django.shortcuts import render, redirect
from .forms import LoginForm
from .forms import CreateUserForm


def home(request):
    return render(request, 'base/home.html')

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


def registerPage(request):
	if request.user.is_authenticated:
		return redirect('home')
	else:
		form = CreateUserForm()
		if request.method == 'POST':
			form = CreateUserForm(request.POST)
			if form.is_valid():
				form.save()
				user = form.cleaned_data.get('username')
				messages.success(request, 'Account was created for ' + user)

				return redirect('login')
			

		context = {'form':form}
		return render(request, 'base/home.html', context)

def coursePage(request):
    return render(request, 'base/try_excercise.html')

@gzip.gzip_page
def mediapipePage(request):
    cam = VideoCamera()
    return StreamingHttpResponse(gen(cam, request), content_type="multipart/x-mixed-replace;boundary=frame")
    
#to capture video class
class VideoCamera(object):
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
    
    