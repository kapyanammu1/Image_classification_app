
from django.contrib import admin
from django.urls import path
from ICT_App.views import classify_image
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', classify_image, name='classify_image'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)