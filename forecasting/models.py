from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator

def profile_pic_upload_path_handler(instance):
    return "avatars/{id}/profile_pic.jpg".format(id=instance.user.id)
def parameters_upload_path_handler(instance, filename):
    return "parameters/{user_id}_{group_id}/parameters.txt".format(user_id=instance.user.id, group_id=instance.group.id)
def timeseries_upload_path_handler(instance, filename):
    return "timeseries/{user_id}_{group_id}/timeseries.txt".format(user_id=instance.user.id, group_id=instance.group.id)

class UserProfile(models.Model):
    user = models.OneToOneField(User, related_name='profile', on_delete=models.CASCADE)
    image = models.ImageField(upload_to=profile_pic_upload_path_handler, blank=True)
    bio = models.CharField(max_length=256, blank=True, null=True)

class Group(models.Model):
    name = models.CharField(max_length=32, null=False, unique=False)
    location = models.CharField(max_length=32, null=False)
    description = models.CharField(max_length=256, null=False, unique=False)

    class Meta:
        unique_together = (('name', 'location'))

class Member(models.Model):
    group = models.ForeignKey(Group, related_name='members', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='members_of', on_delete=models.CASCADE)
    admin = models.BooleanField(default=False)

    class Meta:
        unique_together = (('user', 'group'),)

class Comment(models.Model):
    group = models.ForeignKey(Group, related_name='comments', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='user_comments', on_delete=models.CASCADE)
    description = models.CharField(max_length=256, null=False, unique=False)
    time = models.DateTimeField(auto_now_add=True)

class TimeSeriesData(models.Model):
    group = models.ForeignKey(Group, related_name='timeseries', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='uploaded_timeseries', on_delete=models.CASCADE)
    file = models.FileField(
        upload_to=timeseries_upload_path_handler,
        validators=[FileExtensionValidator(allowed_extensions=['txt'])]
    )
    upload_date = models.DateTimeField(auto_now_add=True)
    description = models.CharField(max_length=256, blank=True)

class ParameterFile(models.Model):
    group = models.ForeignKey(Group, related_name='parameter_files', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='uploaded_parameters', on_delete=models.CASCADE)
    file = models.FileField(
        upload_to=parameters_upload_path_handler,
        validators=[FileExtensionValidator(allowed_extensions=['txt'])],
        blank=True,
        null=True
    )
    upload_date = models.DateTimeField(auto_now_add=True)
    description = models.CharField(max_length=256, blank=True)

class SimulationRun(models.Model):
    MODE_CHOICES = [
        ('F', 'Forward Mode'),
        ('P', 'PSO Calibrator'),
        ('L', 'Latin Hypercube Calibrator'),
        ('M', 'Monte Carlo Calibrator')
    ]

    ERROR_METRIC_CHOICES = [
        ('R', 'RMSE'),
        ('K', 'KGE'),
        ('N', 'NSE')
    ]

    SOLVER_CHOICES = [
        ('E', 'Euler'),
        ('T', 'RK2'),
        ('F', 'RK4'),
        ('C', 'Crank Nicolson')
    ]

    group = models.ForeignKey(Group, related_name='simulations', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='simulation_runs', on_delete=models.CASCADE)
    timeseries = models.ForeignKey(TimeSeriesData, related_name='simulations', on_delete=models.CASCADE)
    parameters = models.ForeignKey(ParameterFile, related_name='forward_simulations', on_delete=models.CASCADE)

    # Basic simulation parameters
    mode = models.CharField(max_length=1, choices=MODE_CHOICES)
    error_metric = models.CharField(max_length=1, choices=ERROR_METRIC_CHOICES)
    solver = models.CharField(max_length=1, choices=SOLVER_CHOICES)

    # Status and results
    status = models.CharField(max_length=20, default='pending')
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    results_path = models.CharField(max_length=512, null=True, blank=True)

class PSOParameters(models.Model):
    simulation = models.OneToOneField(
        SimulationRun,
        related_name='pso_params',
        on_delete=models.CASCADE
    )
    swarm_size = models.IntegerField(default=30)
    phi1 = models.FloatField(default=2.05)
    phi2 = models.FloatField(default=2.05)
    max_iterations = models.IntegerField(default=100)



class LatinParameters(models.Model):
    simulation = models.OneToOneField(
        SimulationRun,
        related_name='latin_params',
        on_delete=models.CASCADE
    )
    num_samples = models.IntegerField(default=100)


class MonteCarloParameters(models.Model):
    simulation = models.OneToOneField(
        SimulationRun,
        related_name='monte_params',
        on_delete=models.CASCADE
    )
    num_iterations = models.IntegerField(default=1000)

