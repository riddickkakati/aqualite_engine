from django.urls import path, include
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register("ml_groups", views.MLGroupViewset)
router.register("ml_members", views.MLMemberViewset)
router.register("ml_comments", views.MLCommentViewset)
router.register("users", views.UserViewSet)
router.register("profile", views.UserProfileViewset)
router.register("ml_analysis", views.MLRunViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path("authenticate/", views.CustomObtainAuthToken.as_view())
]