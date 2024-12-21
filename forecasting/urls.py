from django.urls import path, include
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
# Existing routes
router.register("groups", views.GroupViewset)
router.register("members", views.MemberViewset)
router.register("comments", views.CommentViewset)
router.register("users", views.UserViewSet)
router.register("profile", views.UserProfileViewset)

# New forecasting routes
router.register("timeseries", views.TimeSeriesDataViewSet)
router.register("parameters", views.ParameterFileViewSet)
router.register("simulations", views.SimulationRunViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path("authenticate/", views.CustomObtainAuthToken.as_view())
]
