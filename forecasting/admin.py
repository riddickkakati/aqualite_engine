from django.contrib import admin
from .models import Group, UserProfile, Member, Comment, SimulationRun, TimeSeriesData, ParameterFile, PSOParameter, LatinParameter, MonteCarloParameter, ForwardParameter


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    fields = ('user', 'image', 'is_premium', 'bio')
    list_display = ('id', 'user', 'image')

@admin.register(Group)
class GroupAdmin(admin.ModelAdmin):
    fields = ('name', 'location', 'time', 'description')
    list_display = ('id','name', 'time', 'location', 'description')


@admin.register(Member)
class MemberAdmin(admin.ModelAdmin):
    fields = ('user', 'group', 'admin')
    list_display = ('user', 'group', 'admin')

@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    fields = ('user', 'group', 'description')
    list_display = ('user', 'group', 'description', 'time')


@admin.register(SimulationRun)
class SimulationRunAdmin(admin.ModelAdmin):
    fields = (
         'user', 'group',
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

    list_display = (
        'user', 'group',
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

@admin.register(TimeSeriesData)
class TimeSeriesDataAdmin(admin.ModelAdmin):
    fields = ('group', 'user', 'file', 'description')
    list_display = ('id', 'group', 'user', 'file', 'upload_date', 'description')

@admin.register(ParameterFile)
class ParameterFileAdmin(admin.ModelAdmin):
    fields = ('group', 'user', 'file', 'description')
    list_display = ('id', 'group', 'user', 'file', 'upload_date', 'description')

@admin.register(ForwardParameter)
class ForwardParametersAdmin(admin.ModelAdmin):
    fields = ('group', 'user', 'parameter1', 'parameter2', 'parameter3',
                    'parameter4', 'parameter5', 'parameter6', 'parameter7', 'parameter8')
    list_display =  ('id', 'group', 'user', 'parameter1', 'parameter2', 'parameter3',
                    'parameter4', 'parameter5', 'parameter6', 'parameter7', 'parameter8')

@admin.register(PSOParameter)
class PSOParametersAdmin(admin.ModelAdmin):
    fields = ('simulation','swarm_size', 'phi1', 'phi2', 'max_iterations')
    list_display = ('simulation','swarm_size', 'phi1', 'phi2', 'max_iterations')
@admin.register(LatinParameter)
class LatinParametersAdmin(admin.ModelAdmin):
    fields = ('simulation','num_samples',)
    list_display = ('simulation','num_samples',)

@admin.register(MonteCarloParameter)
class MonteCarloParametersAdmin(admin.ModelAdmin):
    fields = ('simulation','num_iterations',)
    list_display = ('simulation','num_iterations',)
