from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from forecasting.serializers import UserSerializer
from .models import MonitoringRun, MonitoringGroup, MonitoringMember, MonitoringComment

class CommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonitoringComment
        fields = ('user', 'group', 'description', 'time')


class MemberSerializer(serializers.ModelSerializer):
    user = UserSerializer(many=False)

    class Meta:
        model = MonitoringMember
        fields = ('user', 'group', 'admin')


class GroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonitoringGroup
        fields = ('id', 'name', 'location', 'description',)


class GroupFullSerializer(serializers.ModelSerializer):
    monitoring_members = serializers.SerializerMethodField()
    monitoring_comments = serializers.SerializerMethodField()

    class Meta:
        model = MonitoringGroup
        fields = ('id', 'name', 'time', 'location', 'description', 'monitoring_members', 'monitoring_comments')

    def get_monitoring_comments(self, obj):
        monitoring_comments = MonitoringComment.objects.filter(group=obj).order_by('-time')
        serializer = CommentSerializer(monitoring_comments, many=True)
        return serializer.data

    def get_forecasting_members(self, obj):  # Changed name to match your field
        monitoring_members = obj.monitoring_members.all()
        return MemberSerializer(monitoring_members, many=True).data

class MonitoringRunSerializer(serializers.ModelSerializer):
    user = UserSerializer()
    group = GroupSerializer()

    class Meta:
        model = MonitoringRun
        fields = (
            'id', 'user', 'group', 'start_date', 'end_date', 'longitude', 'latitude', 'satellite', 'parameter', 'error_message', 'updated_at', 'status', 'start_time', 'end_time', 'results_path')
