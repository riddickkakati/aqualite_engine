from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from .models import (
    Group, UserProfile, Member, Comment,
    TimeSeriesData, ParameterFile, SimulationRun,
    PSOParameters, LatinParameters, MonteCarloParameters
)
from django.db.models import Sum


class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ('id', 'image', 'bio')


class UserSerializer(serializers.ModelSerializer):
    profile = UserProfileSerializer()

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'profile')
        extra_kwargs = {'password': {'write_only': True, 'required': False}}

    def create(self, validated_data):
        profile_data = validated_data.pop('profile')
        user = User.objects.create_user(**validated_data)
        UserProfile.objects.create(user=user, **profile_data)
        Token.objects.create(user=user)
        return user


class CommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ('user', 'group', 'description', 'time')


class MemberSerializer(serializers.ModelSerializer):
    user = UserSerializer(many=False)

    class Meta:
        model = Member
        fields = ('user', 'group', 'admin')


class GroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = Group
        fields = ('id', 'name', 'location', 'description')


class GroupFullSerializer(serializers.ModelSerializer):
    members = serializers.SerializerMethodField()
    comments = serializers.SerializerMethodField()

    class Meta:
        model = Group
        fields = ('id', 'name', 'location', 'description', 'members', 'comments')

    def get_comments(self, obj):
        comments = Comment.objects.filter(group=obj).order_by('-time')
        serializer = CommentSerializer(comments, many=True)
        return serializer.data


# New serializers for forecasting functionality
class TimeSeriesDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = TimeSeriesData
        fields = ('id', 'group', 'user', 'file', 'upload_date', 'description')
        read_only_fields = ('upload_date',)


class ParameterFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = ParameterFile
        fields = ('id', 'group', 'user', 'file', 'upload_date', 'description')
        read_only_fields = ('upload_date',)


class PSOParametersSerializer(serializers.ModelSerializer):
    class Meta:
        model = PSOParameters
        fields = ('swarm_size', 'phi1', 'phi2', 'max_iterations')


class LatinParametersSerializer(serializers.ModelSerializer):
    class Meta:
        model = LatinParameters
        fields = ('num_samples',)


class MonteCarloParametersSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonteCarloParameters
        fields = ('num_iterations', 'confidence_level')


class SimulationRunSerializer(serializers.ModelSerializer):
    user = UserSerializer()
    group = GroupSerializer()
    timeseries = TimeSeriesDataSerializer()
    parameters = ParameterFileSerializer()
    pso_params = PSOParametersSerializer(required=False)
    latin_params = LatinParametersSerializer(required=False)
    monte_params = MonteCarloParametersSerializer(required=False)

    class Meta:
        model = SimulationRun
        fields = (
            'id', 'user', 'group', 'timeseries', 'parameters',
            'mode', 'error_metric', 'solver', 'status',
            'start_time', 'end_time', 'results_path',
            'pso_params', 'latin_params', 'monte_params'
        )
        read_only_fields = ('start_time', 'end_time', 'status', 'results_path')

    def create(self, validated_data):
        # Extract nested parameters based on mode
        mode = validated_data.get('mode')
        params_data = None
        params_serializer = None

        if mode == 'pso':
            params_data = validated_data.pop('pso_params', None)
            params_serializer = PSOParametersSerializer
        elif mode == 'latin':
            params_data = validated_data.pop('latin_params', None)
            params_serializer = LatinParametersSerializer
        elif mode == 'monte':
            params_data = validated_data.pop('monte_params', None)
            params_serializer = MonteCarloParametersSerializer

        # Create simulation run
        simulation = SimulationRun.objects.create(**validated_data)

        # Create parameters if provided
        if params_data and params_serializer:
            params_serializer().create(dict(simulation=simulation, **params_data))

        return simulation


class GroupForecastSerializer(serializers.ModelSerializer):
    simulations = SimulationRunSerializer(many=True, read_only=True)
    timeseries = TimeSeriesDataSerializer(many=True, read_only=True)
    parameter_files = ParameterFileSerializer(many=True, read_only=True)

    class Meta:
        model = Group
        fields = ('id', 'name', 'location', 'description', 'simulations', 'timeseries', 'parameter_files')