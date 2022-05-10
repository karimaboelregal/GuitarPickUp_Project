from django.contrib import admin

# Register your models here.
from GuitarPickUp.models import Excercise,Feedback,Feedback_details


admin.site.register(Excercise)

admin.site.register(Feedback)
admin.site.register(Feedback_details)