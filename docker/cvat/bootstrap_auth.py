import os

from django.contrib.auth import get_user_model
from rest_framework.authtoken.models import Token


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    username = os.getenv("CVAT_AUTOMATION_USERNAME", "chimp-automation").strip()
    password = os.getenv("CVAT_AUTOMATION_PASSWORD", "change-me-now").strip()
    email = os.getenv("CVAT_AUTOMATION_EMAIL", "chimp-automation@example.local").strip()
    is_superuser = _truthy(os.getenv("CVAT_AUTOMATION_IS_SUPERUSER"), default=True)
    fixed_token = os.getenv("CVAT_AUTOMATION_API_TOKEN", "").strip()

    if not username:
        raise RuntimeError("CVAT_AUTOMATION_USERNAME cannot be empty")
    if not password:
        raise RuntimeError("CVAT_AUTOMATION_PASSWORD cannot be empty")

    User = get_user_model()
    user, created = User.objects.get_or_create(username=username)

    user.email = email
    user.set_password(password)
    user.is_superuser = is_superuser
    user.is_staff = True if is_superuser else user.is_staff
    user.save()

    if fixed_token:
        try:
            token = Token.objects.get(user=user)
            if token.key != fixed_token:
                token.delete()
                token = Token.objects.create(user=user, key=fixed_token)
        except Token.DoesNotExist:
            token = Token.objects.create(user=user, key=fixed_token)
    else:
        token, _ = Token.objects.get_or_create(user=user)

    action = "created" if created else "updated"
    print(f"CVAT automation user {action}: username={user.username} superuser={user.is_superuser}")
    print("CVAT automation API token is ready.")


main()
