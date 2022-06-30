from .models import FacesUser
from django.forms import ModelForm

class FacesUserForm(ModelForm):
    class Meta:
        model = FacesUser
        fields = ['name', 'image']
