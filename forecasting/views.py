from django.contrib.auth.models import User
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.parsers import MultiPartParser, FormParser
from datetime import datetime
from .models import (
    Group, UserProfile, Member, Comment,
    TimeSeriesData, ParameterFile, SimulationRun,
    PSOParameters, LatinParameters, MonteCarloParameters
)
from .serializers import (
    GroupSerializer, GroupFullSerializer, GroupForecastSerializer,
    UserSerializer, UserProfileSerializer, ChangePasswordSerializer,
    MemberSerializer, CommentSerializer, TimeSeriesDataSerializer,
    ParameterFileSerializer, SimulationRunSerializer
)
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAuthenticatedOrReadOnly


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
        # Ensure `get_queryset` always returns a valid queryset
        if self.action in ['list', 'retrieve']:
            return SimulationRun.objects.filter(user=self.request.user.id)
        return SimulationRun.objects.all()

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

    @action(methods=['post'], detail=True)
    def run_simulation(self, request, pk=None):
        simulation = self.get_object()

        # Here you would implement the logic to run your forecasting model
        # This is a placeholder for where you'd call your actual model
        try:
            # Example placeholder for model execution
            # result_path = your_forecasting_model.run(simulation)
            # simulation.set_results_path(result_path)

            return Response({
                'message': 'Simulation started successfully',
                'simulation_id': simulation.id
            }, status=status.HTTP_200_OK)
        except Exception as e:
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
