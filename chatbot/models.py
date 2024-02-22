from django.db import models

# Create your models here.

class DoctorDetails(models.Model):
    doctorname = models.CharField(max_length=100,blank=True,null=True)
    department = models.CharField(max_length=100,blank=True,null=True)
    experience = models.PositiveIntegerField(blank=True,null=True)
    place = models.CharField(max_length=100,blank=True,null=True)
    hospital_name = models.CharField(max_length=100,blank=True,null=True)
    contact_number = models.CharField(max_length=100,blank=True,null=True)
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

class MultipleDoctors(models.Model):
    user = models.CharField(max_length=255, null=True, blank=True)
    department = models.CharField(max_length=255, null=True, blank=True)
    place = models.CharField(max_length=100, null=True, blank=True)
    hospital_name = models.CharField(max_length=100, null=True, blank=True)
    experience = models.PositiveBigIntegerField(null=True, blank=True)