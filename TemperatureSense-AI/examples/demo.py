from smart_collar_temperature import (
    ActivityLevel,
    BodyCondition,
    CoatType,
    DogProfile,
    EnvironmentContext,
    RabiesContext,
    SamplingPolicy,
    SizeClass,
    SleepState,
    TemperatureScheduler,
    assess_environment,
    assess_surface_contact_temperature,
    assess_rabies_context,
)


dog = DogProfile(
    breed="Siberian Husky",
    age_years=4,
    size=SizeClass.LARGE,
    coat=CoatType.HEAVY_DOUBLE,
    body_condition=BodyCondition.IDEAL,
    healthy_sleep_surface_baseline_c=33.1,
)

scheduler = TemperatureScheduler(
    SamplingPolicy(sleeping_environment_interval_s=None)
)

awake = scheduler.schedule(
    now_s=1000,
    bmi270_state=SleepState.AWAKE,
    state_since_s=700,
    last_sht40_s=950,
    last_ntc_s=None,
)
print("Awake schedule:", awake)

environment = assess_environment(
    31.0,
    78.0,
    dog,
    EnvironmentContext(activity=ActivityLevel.ACTIVE, direct_sun=True),
)
print("Environment:", environment.severity.name, environment.range_label)
print("Actions:", *environment.actions, sep="\n- ")

sleep = scheduler.schedule(
    now_s=2000,
    bmi270_state=SleepState.SLEEPING,
    state_since_s=1500,
    last_sht40_s=1950,
    last_ntc_s=1900,
)
print("Sleep schedule:", sleep)

surface = assess_surface_contact_temperature(
    [34.0, 34.1, 34.0, 34.2, 34.1, 34.0],
    dog,
    sleep_confirmed=sleep.sleep_confirmed,
)
print("Surface-contact trend:", surface.severity.name, surface.range_label)

rabies = assess_rabies_context(RabiesContext(), surface)
print("Rabies rule:", rabies.code, "-", rabies.message)
