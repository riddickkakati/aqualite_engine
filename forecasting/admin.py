from django.contrib import admin
from .models import Group, UserProfile, Member, Comment, SimulationRun, TimeSeriesData, ParameterFile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    fields = ('user', 'image', 'is_premium', 'bio')
    list_display = ('id', 'user', 'image')

@admin.register(Group)
class GroupAdmin(admin.ModelAdmin):
    fields = ('name', 'location', 'description')
    list_display = ('id','name', 'location', 'description')


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
        'group', 'user', 'timeseries', 'parameters', 'mode',
        'error_metric', 'solver', 'status'
    )
    list_display = (
        'id', 'group', 'user', 'timeseries', 'parameters', 'mode',
        'error_metric', 'solver', 'status', 'start_time', 'end_time'
    )

@admin.register(TimeSeriesData)
class TimeSeriesDataAdmin(admin.ModelAdmin):
    fields = ('group', 'user', 'file', 'description')
    list_display = ('id', 'group', 'user', 'file', 'upload_date', 'description')

@admin.register(ParameterFile)
class ParameterFileAdmin(admin.ModelAdmin):
    fields = ('group', 'user', 'file', 'description')
    list_display = ('id', 'group', 'user', 'file', 'upload_date', 'description')

