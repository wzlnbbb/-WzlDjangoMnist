from django.shortcuts import render, HttpResponse
from helloapp.forms import ProfileForm
from helloapp.models import Profile
from mnist_test import predict
# Create your views here.


def index(request):
    context = {}
    if request.method == "POST":
        MyProfileForm = ProfileForm(request.POST, request.FILES)
        if MyProfileForm.is_valid():
            profile = Profile()
            # profile.name = MyProfileForm.cleaned_data["name"]
            profile.picture = MyProfileForm.cleaned_data["picture"]
            profile.save()
            pic = Profile.objects.all()
            pic_url = pic[len(pic)-1].picture.url
            return HttpResponse(predict('.'+pic_url))
            # return render(request, 'index.html', {'result': predict('.'+pic_url)})
    else:
        form = ProfileForm()
        context['form'] = form
    return render(request, 'index.html', context)