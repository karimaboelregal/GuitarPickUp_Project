from django.db import models
from django.contrib.auth.models import User
import uuid

from django.db import models
from django.urls.base import reverse

# Create your models here.

class Excercise(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE,null=True,blank=True)
    title = models.CharField(max_length=200) 
    positions = models.TextField(null=True,blank=True)
    path = models.TextField(null=True,blank=True)
    played = models.BooleanField(default=False)

    def __str__(self) -> str:
        return self.title

    class Meta:
        ordering =['played']

class StudentVideo(models.Model):
    user_id = models.ForeignKey(User,on_delete=models.CASCADE,null=True,blank=True)
    Excercise_id = models.ForeignKey(Excercise,on_delete=models.CASCADE,null=True,blank=True)
    feedback_id = models.IntegerField(null=True,blank=True)
    report= models.TextField(null=True,blank=True)
    video_record = models.FileField(null = True,upload_to="records")
    video_Path = models.TextField(null=True,blank=True)

    def __str__(self) -> str:
        return self.title

    class Meta:
        ordering =['user_id']


class Feedback(models.Model):
    video_id = models.ForeignKey(StudentVideo,on_delete=models.CASCADE,null=True,blank=True)
    user_id = models.ForeignKey(User,on_delete=models.CASCADE,null=True,blank=True)
    feedback = models.TextField(null=True,blank=True)
    report= models.TextField(null=True,blank=True)

    def __str__(self) -> str:
        return self.feedback

    class Meta:
        ordering =['video_id']

class Feedback_details(models.Model):
    feedback_id = models.ForeignKey(Feedback,on_delete=models.CASCADE,null=True,blank=True)
    index_class = models.BooleanField(null=True,blank=True)
    middle_class = models.BooleanField(null=True,blank=True)
    ring_class = models.BooleanField(null=True,blank=True)
    pinky_class = models.BooleanField(null=True,blank=True)
    note_played = models.CharField(null = True, blank=True,max_length=5)
    note_class = models.BooleanField(null=True,blank=True)
    
    def __str__(self) -> str:
        return self.feedback_detail

    class Meta:
        ordering =['feedback_id']

# class Record(models.Model):
#     id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
#     voice_record = models.FileField(upload_to="records")
#     language = models.CharField(max_length=50, null=True, blank=True)

#     class Meta:
#         verbose_name = "Record"
#         verbose_name_plural = "Records"

#     def __str__(self):
#         return str(self.id)

#     def get_absolute_url(self):
#         return reverse("core:record_detail", kwargs={"id": str(self.id)})