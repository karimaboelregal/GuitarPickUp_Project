from django.db import models
from django.contrib.auth.models import User

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
    Path = models.TextField(null=True,blank=True)

    def __str__(self) -> str:
        return self.title

    class Meta:
        ordering =['user_id']


class Feedback(models.Model):
    video_id = models.ForeignKey(StudentVideo,on_delete=models.CASCADE,null=True,blank=True)
    feedback = models.TextField(null=True,blank=True)
    report= models.TextField(null=True,blank=True)

    def __str__(self) -> str:
        return self.feedback

    class Meta:
        ordering =['video_id']
