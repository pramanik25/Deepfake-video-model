# detector_app/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings
from .model_utils import predict_single_video

# For basic auth (replace with django.contrib.auth for production)
from django.contrib.auth.models import User # If using Django's User model
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required


def index_view(request):
    return render(request, 'detector_app/index.html')

@login_required(login_url='detector_app:login_view') # Redirect to login if not authenticated
def predict_video_view(request):
    if request.method == 'POST':
        video_file = request.FILES.get('videoFile')
        if not video_file:
            return JsonResponse({'error': 'No video file provided'}, status=400)

        allowed_video_types = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/avi', 'video/webm']
        if video_file.content_type not in allowed_video_types:
            return JsonResponse({'error': f'Unsupported file type: {video_file.content_type}.'}, status=400)

        max_size = 100 * 1024 * 1024 # 100 MB
        if video_file.size > max_size:
            return JsonResponse({'error': f'File is too large. Max size is {max_size // (1024*1024)}MB.'}, status=400)

        media_dir = settings.MEDIA_ROOT
        if not os.path.exists(media_dir):
            try:
                os.makedirs(media_dir)
            except OSError as e:
                print(f"Error creating media directory {media_dir}: {e}")
                return JsonResponse({'error': 'Server configuration error (media storage).'}, status=500)
            
        fs = FileSystemStorage(location=media_dir)
        filename = fs.save(video_file.name, video_file)
        video_file_system_path = os.path.join(media_dir, filename)
        print(f"Video saved to: {video_file_system_path}")

        result = predict_single_video(video_file_system_path)
        print(f"Prediction result: {result}")

        try:
            if os.path.exists(video_file_system_path):
                os.remove(video_file_system_path)
                print(f"Removed temporary file: {video_file_system_path}")
            else:
                print(f"Warning: Temporary file not found for removal: {video_file_system_path}")
        except Exception as e:
            print(f"Error removing temporary file {video_file_system_path}: {e}")

        if 'error' in result:
             return JsonResponse(result, status=500 if "Model not loaded" not in result['error'] else 503)
        return JsonResponse(result)

    return JsonResponse({'error': 'Invalid request method. Only POST allowed.'}, status=405)

# --- Authentication Views ---
def login_view(request):
    if request.user.is_authenticated:
        return redirect('detector_app:index')
    return render(request, 'detector_app/login.html')

def login_action_view(request):
    if request.method == 'POST':
        email = request.POST.get('email') # Using email as username
        password = request.POST.get('password')
        
        # Django's authenticate expects a username. If your User model uses email as USERNAME_FIELD:
        # user = authenticate(request, username=email, password=password)
        # If not, you might need to fetch user by email first:
        try:
            user_obj = User.objects.get(email=email)
            user = authenticate(request, username=user_obj.username, password=password)
        except User.DoesNotExist:
            user = None

        if user is not None:
            auth_login(request, user)
            # Redirect to a success page or the main page
            # Check for 'next' parameter for redirection after login
            next_url = request.GET.get('next', None)
            if next_url:
                return redirect(next_url)
            return redirect('detector_app:index') 
        else:
            # Invalid login
            return render(request, 'detector_app/login.html', {'error_message': 'Invalid email or password.'})
    return redirect('detector_app:login_view')


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('detector_app:index')
    return render(request, 'detector_app/sign_up.html')

def signup_action_view(request):
    if request.method == 'POST':
        full_name = request.POST.get('full_name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        # username could be derived from email or be the email itself if configured
        username = email.split('@')[0] # Simple way, ensure uniqueness

        # Basic validation (add more)
        if not all([full_name, email, password]):
            return render(request, 'detector_app/sign_up.html', {'error_message': 'All fields are required.'})
        if User.objects.filter(email=email).exists():
            return render(request, 'detector_app/sign_up.html', {'error_message': 'Email already exists.'})
        if User.objects.filter(username=username).exists():
            # Handle username collision, e.g., append a number or ask user for a username
            return render(request, 'detector_app/sign_up.html', {'error_message': 'A user with a similar username already exists. Try a different email.'})

        try:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.first_name = full_name.split(' ')[0]
            if ' ' in full_name:
                 user.last_name = ' '.join(full_name.split(' ')[1:])
            user.save()
            # Log the user in directly after signup or redirect to login
            auth_login(request, user)
            return redirect('detector_app:index')
            # return render(request, 'detector_app/sign_up.html', {'success_message': 'Account created! Please login.'})
        except Exception as e:
            print(f"Error creating user: {e}")
            return render(request, 'detector_app/sign_up.html', {'error_message': f'Could not create account: {e}'})
            
    return redirect('detector_app:signup_view')

def logout_view(request):
    auth_logout(request)
    return redirect('detector_app:login_view') # Redirect to login page after logout