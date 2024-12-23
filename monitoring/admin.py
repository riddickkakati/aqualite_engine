from django.contrib import admin
from .models import MonitoringRun

@admin.register(MonitoringRun)
class SimulationRunAdmin(admin.ModelAdmin):
    fields = ('user', 'group', 'start_date', 'end_date', 'longitude', 'latitude', 'satellite', 'parameter', 'error_message', 'updated_at', 'status', 'start_time', 'end_time')

    list_display = ('user', 'group', 'start_date', 'end_date', 'longitude', 'latitude', 'satellite', 'parameter', 'error_message', 'updated_at', 'status', 'start_time', 'end_time')
