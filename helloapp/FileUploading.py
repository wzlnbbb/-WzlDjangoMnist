def HandleUpLoadedFile(File):
    with open('UpLoadedFile/name.txt','wb+') as destination:
        for chunk in File.chunks():
            destination.write(chunk)