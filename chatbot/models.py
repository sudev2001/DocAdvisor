from django.db import models

# Create your models here.

class DoctorDetails(models.Model):
    doctorname = models.CharField(max_length=100,blank=True,null=True)
    department = models.CharField(max_length=100,blank=True,null=True)
    experience = models.PositiveIntegerField(blank=True,null=True)
    positive_count = models.PositiveIntegerField(blank=True,null=True)
    negative_count = models.PositiveIntegerField(blank=True,null=True)
    neutral_count = models.PositiveIntegerField(blank=True,null=True)
    total_reviews = models.PositiveIntegerField(blank=True,null=True)
    rank_math = models.FloatField(blank=True,null=True)
    rank = models.PositiveIntegerField(blank=True,null=True)

class Review(models.Model):
    department = models.CharField(max_length=100,blank=True,null=True)
    doctor_id = models.ForeignKey(DoctorDetails,on_delete=models.CASCADE)
    review = models.TextField(null=True,blank=True)