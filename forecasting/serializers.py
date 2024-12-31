from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from .models import (
    ForecastingGroup, UserProfile, ForecastingMember, ForecastingComment,
    TimeSeriesData, ParameterFile, SimulationRun,
    PSOParameter, LatinParameter, MonteCarloParameter, ForwardParameter, ParameterRangesFile, UserValidationFile
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
        model = ForecastingComment
        fields = ('user', 'group', 'description', 'time')


class MemberSerializer(serializers.ModelSerializer):
    user = UserSerializer(many=False)

    class Meta:
        model = ForecastingMember
        fields = ('user', 'group', 'admin', 'time')


class GroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastingGroup
        fields = ('id', 'name', 'location', 'description',)


class GroupFullSerializer(serializers.ModelSerializer):
    forecasting_members = serializers.SerializerMethodField()
    forecasting_comments = serializers.SerializerMethodField()

    class Meta:
        model = ForecastingGroup
        fields = ('id', 'name', 'time', 'location', 'description', 'forecasting_members', 'forecasting_comments')

    def get_forecasting_comments(self, obj):
        forecasting_comments = ForecastingComment.objects.filter(group=obj).order_by('-time')
        serializer = CommentSerializer(forecasting_comments, many=True)
        return serializer.data

    def get_forecasting_members(self, obj):  # Changed name to match your field
        forecasting_members = obj.forecasting_members.all()
        return MemberSerializer(forecasting_members, many=True).data


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

class ParameterRangesFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = ParameterRangesFile
        fields = ('id', 'group', 'user', 'file', 'upload_date', 'description')

class UserValidationFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserValidationFile
        fields = ('id', 'group', 'user', 'file', 'upload_date', 'description')

class ForwardParametersSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForwardParameter
        fields = ('id', 'group', 'user', 'parameter1', 'parameter2', 'parameter3',
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
    parameter_ranges = ParameterRangesFileSerializer(required=False)
    user_validation = UserValidationFileSerializer(required=False)
    pso_params = PSOParametersSerializer(required=False)
    latin_params = LatinParametersSerializer(required=False)
    monte_params = MonteCarloParametersSerializer(required=False)

    class Meta:
        model = SimulationRun
        fields = (
            'id', 'user', 'group', 'timeseries', 'parameters_file', 'parameters_forward', 'parameter_ranges', 'user_validation',
            # Basic simulation parameters
            'interpolate', 'n_data_interpolate', 'validation_required', 'core',
            'depth', 'compiler', 'CFL', 'databaseformat', 'computeparameterranges',
            'computeparameters', 'model', 'mode', 'method', 'optimizer',
            'forward_options', 'error_metric', 'solver', 'log_flag',
            'resampling_frequency_days', 'resampling_frequency_weeks',
            'email_send', 'email_list',
            # Status and results
            'error_message', 'updated_at', 'status', 'start_time', 'end_time', 'results_path',
            # Optional parameter sets
            'pso_params', 'latin_params', 'monte_params'
        )
        read_only_fields = ('start_time', 'end_time', 'status', 'results_path')

    def to_representation(self, instance):
        representation = super().to_representation(instance)

        if instance.computeparameterranges == True:
            representation.pop('parameter_ranges', None)

        if instance.model == 'W':
            representation.pop('CFL', None)

        if instance.validation_required == False:
            representation.pop('user_validation', None)

        if instance.mode == 'F':
            representation.pop('parameter_ranges', None)
            representation.pop('pso_params', None)
            representation.pop('latin_params', None)
            representation.pop('monte_params', None)
            representation.pop('depth', None)

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
        model = ForecastingGroup
        fields = ('id', 'name', 'location', 'description', 'simulations', 'timeseries', 'parameter_files')