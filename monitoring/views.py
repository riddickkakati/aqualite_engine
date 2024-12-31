from django.contrib.auth.models import User
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.authtoken.views import ObtainAuthToken
from django.utils import timezone
import multiprocessing as mp
from .models import (
    MonitoringRun,
)

from forecasting.models import (
    Group, UserProfile, Member, Comment
)

from forecasting.serializers import (
    UserSerializer, UserProfileSerializer, ChangePasswordSerializer
)

from .serializers import MonitoringRunSerializer, GroupSerializer, GroupFullSerializer, MemberSerializer, CommentSerializer
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAuthenticatedOrReadOnly
from .Air2water_GEE import Air2water_monit


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

                member = Member.objects.create(group=group, user=user, admin=request.data.get('admin', False))
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
class MonitoringRunViewSet(viewsets.ModelViewSet):
    queryset = MonitoringRun.objects.all()
    serializer_class = MonitoringRunSerializer
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def get_queryset(self):
        if self.action in ['list', 'retrieve']:
            return MonitoringRun.objects.filter(user=self.request.user.id)
        return MonitoringRun.objects.all()

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

    @staticmethod
    def run_monitoring_process(monitoring_id):
        """
        Static method to run the simulation in a separate process
        """
        monitoring = None
        try:
            monitoring = MonitoringRun.objects.get(pk=monitoring_id)

            Run = Air2water_monit(start_date='2022-03-01', end_date='2022-12-15', lat=10.683, long=45.667, cc=7,
                                  satellite=2, variable=2)
            thumb = Run.run()

            variable=0

            if monitoring.parameter == "C":
                variable = 1,
            elif monitoring.parameter == "T":
                variable = 2,
            elif monitoring.parameter == "D":
                variable = 3

            # Initialize model with correct parameters
            model = Air2water_monit(
                start_date=monitoring.start_date,
                end_date=monitoring.end_date,
                long=monitoring.longitude,
                lat=monitoring.latitude,
                cc=7,
                satellite=2 if monitoring.satellite == "S" else 1,
                variable=variable
            )

            # Run the simulation
            num_missing_col3 = model.run()

            # Update simulation with results
            monitoring.status = "completed"
            monitoring.results_path = num_missing_col3
            monitoring.save()

        except MonitoringRun.DoesNotExist:
            # Handle case where simulation object doesn't exist
            print(f"Simulation with id {monitoring_id} not found")
            return

        except Exception as e:
            error_message = str(e)
            print(f"Error in simulation {monitoring_id}: {error_message}")

            if monitoring:
                monitoring.status = "failed"
                monitoring.error_message = error_message[:500]  # Truncate if too long
                monitoring.save()

    @action(methods=['post'], detail=True)
    def run_monitoring(self, request, pk=None):
        monitoring = self.get_object()

        # Check if simulation belongs to user
        if monitoring.user.id != request.user.id:
            return Response({"detail": "You cannot run other users' simulations."},
                            status=status.HTTP_403_FORBIDDEN)

        # Check if simulation is already running
        if monitoring.status == "running":
            # Double check if it's been running for too long (e.g., 24 hours)
            # This helps recover from failed runs that didn't update status
            time_threshold = timezone.now() - timezone.timedelta(minutes=5)
            if monitoring.updated_at < time_threshold:
                monitoring.status = "failed"
                monitoring.error_message = "Simulation timed out"
                monitoring.save()
            else:
                return Response({
                    'message': 'Simulation is already running'
                }, status=status.HTTP_400_BAD_REQUEST)

        # Update simulation status to running
        monitoring.status = "running"
        monitoring.error_message = None  # Clear any previous error message
        monitoring.updated_at = timezone.now()
        monitoring.save()

        # Start the simulation in a separate process
        p = mp.Process(
            target=MonitoringRunViewSet.run_monitoring_process,
            args=(monitoring.id,)
        )
        p.daemon = True  # Make process exit when main program exits
        p.start()

        return Response({
            'message': 'Simulation submitted successfully',
            'simulation_id': monitoring.id
        }, status=status.HTTP_200_OK)

    @action(methods=['get'], detail=True)
    def check_status(self, request, pk=None):
        monitoring = self.get_object()

        if monitoring.user.id != request.user.id:
            return Response({"detail": "You cannot delete other users' data."}, status=status.HTTP_403_FORBIDDEN)

        response_data = {
            'status': monitoring.status,
            'map_path': monitoring.results_path
        }

        # Include error message if simulation failed
        if monitoring.status == "failed" and hasattr(monitoring, 'error_message'):
            response_data['error_message'] = monitoring.error_message

        return Response(response_data)
