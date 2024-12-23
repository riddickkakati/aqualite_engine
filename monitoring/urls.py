from django.urls import path, include
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register("compute", views.MonitoringRunViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path("authenticate/", views.CustomObtainAuthToken.as_view())
]
