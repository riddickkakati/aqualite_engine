from django.contrib.auth.models import User
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.parsers import MultiPartParser, FormParser
from .air2water.Air2water import Air2water_OOP
from django.conf import settings
from datetime import datetime
from .models import (
    Group, UserProfile, Member, Comment,
    TimeSeriesData, ParameterFile, SimulationRun,
    PSOParameter, LatinParameter, MonteCarloParameter
)
from .serializers import (
    GroupSerializer, GroupFullSerializer, GroupForecastSerializer,
    UserSerializer, UserProfileSerializer, ChangePasswordSerializer,
    MemberSerializer, CommentSerializer, TimeSeriesDataSerializer,
    ParameterFileSerializer, SimulationRunSerializer
)
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAuthenticatedOrReadOnly
import os


# Existing ViewSets remain unchanged
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    authentication_classes = (TokenAuthentication,)

    def get_queryset(self):
        # Ensure `get_queryset` always returns a valid queryset
        if self.action in ['list', 'retrieve']:
            return User.objects.filter(id=self.request.user.id)
        return User.objects.all()

    def update(self, request, *args, **kwargs):
        # Restrict updates to the current user only
        if kwargs['pk'] != str(request.user.id):
            return Response({"detail": "You cannot update other users' data."}, status=status.HTTP_403_FORBIDDEN)
        return super().update(request, *args, **kwargs)

    def partial_update(self, request, *args, **kwargs):
        # Restrict partial updates to the current user only
        if kwargs['pk'] != str(request.user.id):
            return Response({"detail": "You cannot update other users' data."}, status=status.HTTP_403_FORBIDDEN)
        return super().partial_update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        # Retrieve the object to be deleted
        instance = self.get_object()

        # Check if the instance belongs to the authenticated user
        if instance.id != request.user.id:
            return Response({"detail": "You cannot delete other users' data."}, status=status.HTTP_403_FORBIDDEN)

        # Proceed to delete the instance
        return super().destroy(request, *args, **kwargs)

    def get_permissions(self):
        # Ensure all actions require authentication
        if self.action in ['list', 'retrieve', 'update', 'partial_update']:
            return [IsAuthenticated()]
        return super().get_permissions()

    @action(methods=['PUT'], detail=True, serializer_class=ChangePasswordSerializer,
            permission_classes=[IsAuthenticated])
    def change_pass(self, request, pk):
        user = User.objects.get(pk=pk)
        serializer = ChangePasswordSerializer(data=request.data)

        if serializer.is_valid():
            if not user.check_password(serializer.data.get('old_password')):
                return Response({'message': 'Wrong old password'}, status=status.HTTP_400_BAD_REQUEST)
            user.set_password(serializer.data.get('new_password'))
            user.save()
            return Response({'message': 'Password Updated'}, status=status.HTTP_200_OK)


class CommentViewset(viewsets.ModelViewSet):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer


class UserProfileViewset(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer


class GroupViewset(viewsets.ModelViewSet):
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticatedOrReadOnly,)

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        if request.query_params.get('forecast_data', False):
            serializer = GroupForecastSerializer(instance, many=False, context={'request': request})
        else:
            serializer = GroupFullSerializer(instance, many=False, context={'request': request})
        return Response(serializer.data)


class MemberViewset(viewsets.ModelViewSet):
    queryset = Member.objects.all()
    serializer_class = MemberSerializer

    @action(methods=['post'], detail=False)
    def join(self, request):
        if 'group' in request.data and 'user' in request.data:
            try:
                group = Group.objects.get(id=request.data['group'])
                user = User.objects.get(id=request.data['user'])

                member = Member.objects.create(group=group, user=user, admin=False)
                serializer = MemberSerializer(member, many=False)
                response = {'message': 'Joined group', 'results': serializer.data}
                return Response(response, status=status.HTTP_200_OK)
            except:
                response = {'message': 'Cannot join'}
                return Response(response, status=status.HTTP_400_BAD_REQUEST)
        else:
            response = {'message': 'Wrong params'}
            return Response(response, status=status.HTTP_400_BAD_REQUEST)

    @action(methods=['post'], detail=False)
    def leave(self, request):
        if 'group' in request.data and 'user' in request.data:
            try:
                group = Group.objects.get(id=request.data['group'])
                user = User.objects.get(id=request.data['user'])

                member = Member.objects.get(group=group, user=user)
                member.delete()
                response = {'message': 'Left group'}
                return Response(response, status=status.HTTP_200_OK)
            except:
                response = {'message': 'Group, user or member not found'}
                return Response(response, status=status.HTTP_400_BAD_REQUEST)
        else:
            response = {'message': 'Wrong params'}
            return Response(response, status=status.HTTP_400_BAD_REQUEST)


class CustomObtainAuthToken(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        response = super(CustomObtainAuthToken, self).post(request, *args, **kwargs)
        token = Token.objects.get(key=response.data['token'])
        user = User.objects.get(id=token.user_id)
        userSerilizer = UserSerializer(user, many=False)
        return Response({'token': token.key, 'user': userSerilizer.data})


# New ViewSets for forecasting functionality
class TimeSeriesDataViewSet(viewsets.ModelViewSet):
    queryset = TimeSeriesData.objects.all()
    serializer_class = TimeSeriesDataSerializer
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)
    parser_classes = (MultiPartParser, FormParser)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def get_queryset(self):
        # Ensure `get_queryset` always returns a valid queryset
        if self.action in ['list', 'retrieve']:
            return TimeSeriesData.objects.filter(user=self.request.user.id)
        return TimeSeriesData.objects.all()

    def update(self, request, *args, **kwargs):
        instance = self.get_object()

        # Check if the instance belongs to the authenticated user
        if instance.id != request.user.id:
            return Response({"detail": "You cannot update other users' data."}, status=status.HTTP_403_FORBIDDEN)
        return super().update(request, *args, **kwargs)

    def partial_update(self, request, *args, **kwargs):
        # Restrict partial updates to the current user only
        instance = self.get_object()

        # Check if the instance belongs to the authenticated user
        if instance.id != request.user.id:
            return Response({"detail": "You cannot update other users' data."}, status=status.HTTP_403_FORBIDDEN)
        return super().partial_update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        # Retrieve the object to be deleted
        instance = self.get_object()

        # Check if the instance belongs to the authenticated user
        if instance.id != request.user.id:
            return Response({"detail": "You cannot delete other users' data."}, status=status.HTTP_403_FORBIDDEN)

        # Proceed to delete the instance
        return super().destroy(request, *args, **kwargs)


class ParameterFileViewSet(viewsets.ModelViewSet):
    queryset = ParameterFile.objects.all()
    serializer_class = ParameterFileSerializer
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)
    parser_classes = (MultiPartParser, FormParser)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def get_queryset(self):
        # Ensure `get_queryset` always returns a valid queryset
        if self.action in ['list', 'retrieve']:
            return TimeSeriesData.objects.filter(user=self.request.user.id)
        return TimeSeriesData.objects.all()

    def update(self, request, *args, **kwargs):
        instance = self.get_object()

        # Check if the instance belongs to the authenticated user
        if instance.id != request.user.id:
            return Response({"detail": "You cannot update other users' data."}, status=status.HTTP_403_FORBIDDEN)
        return super().update(request, *args, **kwargs)

    def partial_update(self, request, *args, **kwargs):
        # Restrict partial updates to the current user only
        instance = self.get_object()

        # Check if the instance belongs to the authenticated user
        if instance.id != request.user.id:
            return Response({"detail": "You cannot update other users' data."}, status=status.HTTP_403_FORBIDDEN)
        return super().partial_update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        # Retrieve the object to be deleted
        instance = self.get_object()

        # Check if the instance belongs to the authenticated user
        if instance.id != request.user.id:
            return Response({"detail": "You cannot delete other users' data."}, status=status.HTTP_403_FORBIDDEN)

        # Proceed to delete the instance
        return super().destroy(request, *args, **kwargs)


class SimulationRunViewSet(viewsets.ModelViewSet):
    queryset = SimulationRun.objects.all()
    serializer_class = SimulationRunSerializer
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def get_queryset(self):
        if self.action in ['list', 'retrieve']:
            return SimulationRun.objects.filter(user=self.request.user.id)
        return SimulationRun.objects.all()

    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.user.id != request.user.id:
            return Response({"detail": "You cannot update other users' data."},
                            status=status.HTTP_403_FORBIDDEN)
        return super().update(request, *args, **kwargs)

    def partial_update(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.user.id != request.user.id:
            return Response({"detail": "You cannot update other users' data."},
                            status=status.HTTP_403_FORBIDDEN)
        return super().partial_update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.user.id != request.user.id:
            return Response({"detail": "You cannot delete other users' data."},
                            status=status.HTTP_403_FORBIDDEN)
        return super().destroy(request, *args, **kwargs)

    @action(methods=['post'], detail=True)
    def run_simulation(self, request, pk=None):
        simulation = self.get_object()

        # Check if simulation belongs to user
        if simulation.user.id != request.user.id:
            return Response({"detail": "You cannot run other users' simulations."},
                            status=status.HTTP_403_FORBIDDEN)

        # Check if simulation is already running
        if simulation.status == "running":
            return Response({
                'message': 'Simulation is already running'
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Update simulation status to running and save start time
            simulation.status = "running"
            simulation.save()

            # Get PSO parameters
            pso_params = simulation.pso_params


            # Build paths for required files
            base_dir = os.path.dirname(simulation.timeseries.file.path)
            results_dir = os.path.join(base_dir, f'results_{simulation.id}')
            os.makedirs(results_dir, exist_ok=True)


            # Initialize model with correct parameters matching __init__
            model = Air2water_OOP(
                # Required parameters from HTTP response
                user_id= simulation.user.id,
                group_id= simulation.group.id,
                interpolate=simulation.interpolate,
                n_data_interpolate=simulation.n_data_interpolate,
                validation_required=simulation.validation_required,
                model="air2water" if simulation.model == "W" else "air2stream",
                core=simulation.core,
                depth=simulation.depth,
                db_file=f"{settings.MEDIA_ROOT}/parameters/{simulation.user.id}_{simulation.group.id}/calibration.db",
                results_file_name= f"{settings.MEDIA_ROOT}/parameters/{simulation.user.id}_{simulation.group.id}/results.db",
                # PSO parameters from pso_params
                swarmsize=pso_params.swarm_size,
                phi1=pso_params.phi1,
                phi2=pso_params.phi2,
                maxiter=pso_params.max_iterations,
                omega=pso_params.omega,

                # Method and mode parameters
                method="SpotPY" if simulation.method == "S" else "PYCUP",
                mode="calibration" if simulation.mode == "C" else "validation",

                # Error metrics and optimization parameters
                error="RMSE" if simulation.error_metric == "R" else "KGE",
                optimizer="PSO" if simulation.optimizer == "P" else "SCE-UA",

                # Technical parameters
                solver="cranknicolson" if simulation.solver == "C" else "explicit",
                compiler="fortran" if simulation.compiler == "F" else "C",
                CFL=simulation.CFL,
                databaseformat="custom" if simulation.databaseformat == "C" else "standard",

                # Computation flags
                computeparametersranges="Yes" if simulation.computeparameterranges else "No",
                computeparameters="Yes" if simulation.computeparameters else "No",

                # File paths
                parameter_ranges=simulation.parameter_ranges_file.path if simulation.parameter_ranges_file else None,
                forward_parameters=None,  # Add if needed
                air2waterusercalibrationpath=simulation.timeseries.file.path,
                air2streamusercalibrationpath=simulation.timeseries.file.path,
                uservalidationpath=simulation.uservalidationpath,

                # Additional parameters
                log_flag=1 if simulation.log_flag else 0,
                resampling_frequency_days=simulation.resampling_frequency_days,
                resampling_frequency_weeks=simulation.resampling_frequency_weeks,
                email_send=1 if simulation.email_send else 0,
                email_list=simulation.email_list.split(',') if simulation.email_list else []
            )

            # Run the simulation
            num_missing_col3 = model.run()

            # Update simulation with results
            simulation.status = "completed"
            #simulation.end_time = timezone.now()
            simulation.results_path = results_dir
            simulation.save()

            return Response({
                'message': 'Simulation completed successfully',
                'simulation_id': simulation.id,
                'num_missing_col3': num_missing_col3,
                'results_path': simulation.results_path
            }, status=status.HTTP_200_OK)

        except Exception as e:
            # Update simulation status to failed
            simulation.status = "failed"
            #simulation.end_time = timezone.now()
            simulation.save()

            return Response({
                'message': f'Error running simulation: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)

    @action(methods=['get'], detail=True)
    def check_status(self, request, pk=None):
        simulation = self.get_object()
        return Response({
            'status': simulation.status,
            'results_path': simulation.results_path
        })
