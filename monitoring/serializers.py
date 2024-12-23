from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from forecasting.serializers import UserSerializer, GroupSerializer
from .models import MonitoringRun

class MonitoringRunSerializer(serializers.ModelSerializer):
    user = UserSerializer()
    group = GroupSerializer()

    class Meta:
        model = MonitoringRun
        fields = (
            'id', 'user', 'group', 'start_date', 'end_date', 'longitude', 'latitude', 'satellite', 'parameter', 'error_message', 'updated_at', 'status', 'start_time', 'end_time', 'results_path',)