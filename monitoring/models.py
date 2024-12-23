from django.db import models
from django.contrib.auth.models import User
from forecasting.models import Group

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

    group = models.ForeignKey(Group, related_name='monitoring_group', on_delete=models.CASCADE)
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
