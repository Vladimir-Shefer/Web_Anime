from django import forms

class ImageForm(forms.Form):
   correctness = forms.BooleanField(required= False)
   picture = forms.ImageField()