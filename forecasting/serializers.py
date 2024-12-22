from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from .models import (
    Group, UserProfile, Member, Comment,
    TimeSeriesData, ParameterFile, SimulationRun,
    PSOParameter, LatinParameter, MonteCarloParameter, ForwardParameter
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
        fields = ('group', 'user', 'file', 'upload_date', 'description')
        read_only_fields = ('upload_date',)

class ForwardParametersSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForwardParameter
        fields = ('group', 'user', 'parameter1', 'parameter2', 'parameter3',
                    'parameter4', 'parameter5', 'parameter6', 'parameter7', 'parameter8')


class PSOParametersSerializer(serializers.ModelSerializer):
    class Meta:
        model = PSOParameter
        fields = ('simulation','swarm_size', 'phi1', 'phi2', 'max_iterations')


class LatinParametersSerializer(serializers.ModelSerializer):
    class Meta:
        model = LatinParameter
        fields = ('simulation','num_samples',)


class MonteCarloParametersSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonteCarloParameter
        fields = ('simulation','num_iterations')


class SimulationRunSerializer(serializers.ModelSerializer):
    user = UserSerializer()
    group = GroupSerializer()
    timeseries = TimeSeriesDataSerializer()
    parameters_file = ParameterFileSerializer(required=False)
    parameters_forward = ForwardParametersSerializer(required=False)
    pso_params = PSOParametersSerializer(required=False)
    latin_params = LatinParametersSerializer(required=False)
    monte_params = MonteCarloParametersSerializer(required=False)

    class Meta:
        model = SimulationRun
        fields = (
            'id', 'user', 'group', 'timeseries', 'parameters_file', 'parameters_forward', 'parameter_ranges_file', 'uservalidationpath',
            # Basic simulation parameters
            'interpolate', 'n_data_interpolate', 'validation_required', 'core',
            'depth', 'compiler', 'CFL', 'databaseformat', 'computeparameterranges',
            'computeparameters', 'model', 'mode', 'method', 'optimizer',
            'forward_options', 'error_metric', 'solver', 'log_flag',
            'resampling_frequency_days', 'resampling_frequency_weeks',
            'email_send', 'email_list',
            # Status and results
            'status', 'start_time', 'end_time', 'results_path',
            # Optional parameter sets
            'pso_params', 'latin_params', 'monte_params'
        )
        read_only_fields = ('start_time', 'end_time', 'status', 'results_path')

    def to_representation(self, instance):
        representation = super().to_representation(instance)

        if instance.mode == 'F':
            representation.pop('pso_params', None)
            representation.pop('latin_params', None)
            representation.pop('monte_params', None)

            if instance.forward_options == 'U':
                representation.pop('parameters_forward', None)

            elif instance.forward_options == 'W':
                representation.pop('parameters_file', None)

        elif instance.mode == 'C':
            if instance.optimizer == 'P':
                representation.pop('latin_params', None)
                representation.pop('monte_params', None)
                representation.pop('parameters_forward', None)
                representation.pop('parameters_file', None)
                representation.pop('forward_options', None)

            elif instance.optimizer == 'L':
                representation.pop('pso_params', None)
                representation.pop('monte_params', None)
                representation.pop('parameters_forward', None)
                representation.pop('parameters_file', None)
                representation.pop('forward_options', None)

            elif instance.optimizer == 'M':
                representation.pop('pso_params', None)
                representation.pop('latin_params', None)
                representation.pop('parameters_forward', None)
                representation.pop('parameters_file', None)
                representation.pop('forward_options', None)

        return representation



class GroupForecastSerializer(serializers.ModelSerializer):
    simulations = SimulationRunSerializer(many=True, read_only=True)
    timeseries = TimeSeriesDataSerializer(many=True, read_only=True)
    parameter_files = ParameterFileSerializer(many=True, read_only=True)

    class Meta:
        model = Group
        fields = ('id', 'name', 'location', 'description', 'simulations', 'timeseries', 'parameter_files')