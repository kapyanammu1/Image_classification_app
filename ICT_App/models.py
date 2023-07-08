from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    classification = models.JSONField(null=True, blank=True)

    def __str__(self):
        return self.image.name
