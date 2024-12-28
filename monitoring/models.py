from django.db import models
from django.contrib.auth.models import User

class MonitoringGroup(models.Model):
    name = models.CharField(max_length=32, null=False, unique=False)
    location = models.CharField(max_length=32, null=False)
    description = models.CharField(max_length=256, null=False, unique=False)
    time = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = (('name', 'location'))

class MonitoringMember(models.Model):
    group = models.ForeignKey(MonitoringGroup, related_name='monitoring_members', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='monitoring_members_of', on_delete=models.CASCADE)
    admin = models.BooleanField(default=False)
    time = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = (('user', 'group'),)

class MonitoringComment(models.Model):
    group = models.ForeignKey(MonitoringGroup, related_name='monitoring_comments', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='monitoring_user_comments', on_delete=models.CASCADE)
    description = models.CharField(max_length=256, null=False, unique=False)
    time = models.DateTimeField(auto_now_add=True)

class MonitoringRun(models.Model):

    SATELLITE_TILE = [
        ('L', 'Landsat'),
        ('S', 'Sentinel')
    ]

    PARAMETER = [
        ('C', 'CHLA'),
        ('T', 'TURBIDITY'),
        ('D', 'DO')
    ]

    group = models.ForeignKey(MonitoringGroup, related_name='monitoring_group', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='monitoring_runs', on_delete=models.CASCADE)
    start_date = models.DateField(null=False, blank=False)
    end_date = models.DateField(null=False, blank=False)
    longitude = models.FloatField(null=False, blank=False)
    latitude = models.FloatField(null=False, blank=False)
    satellite = models.CharField(max_length=1, choices=SATELLITE_TILE)
    parameter = models.CharField(max_length=1, choices=PARAMETER)
    updated_at = models.DateTimeField(auto_now=True)
    error_message = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=20, default='pending')
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    results_path = models.CharField(max_length=512, null=True, blank=True)
