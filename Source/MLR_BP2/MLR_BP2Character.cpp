// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

#include "MLR_BP2Character.h"
#include "HeadMountedDisplayFunctionLibrary.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Components/InputComponent.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/Controller.h"
#include "GameFramework/SpringArmComponent.h"
#include "JsonObjectConverter.h"

//////////////////////////////////////////////////////////////////////////
// AMLR_BP2Character

AMLR_BP2Character::AMLR_BP2Character()
{
	// Set size for collision capsule
	GetCapsuleComponent()->InitCapsuleSize(42.f, 96.0f);

	// set our turn rates for input
	BaseTurnRate = 45.f;
	BaseLookUpRate = 45.f;

	// Don't rotate when the controller rotates. Let that just affect the camera.
	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;

	// Configure character movement
	GetCharacterMovement()->bOrientRotationToMovement = true; // Character moves in the direction of input...	
	GetCharacterMovement()->RotationRate = FRotator(0.0f, 540.0f, 0.0f); // ...at this rotation rate
	GetCharacterMovement()->JumpZVelocity = 600.f;
	GetCharacterMovement()->AirControl = 0.2f;

	// Create a camera boom (pulls in towards the player if there is a collision)
	CameraBoom = CreateDefaultSubobject<USpringArmComponent>(TEXT("CameraBoom"));
	CameraBoom->SetupAttachment(RootComponent);
	CameraBoom->TargetArmLength = 300.0f; // The camera follows at this distance behind the character	
	CameraBoom->bUsePawnControlRotation = true; // Rotate the arm based on the controller

	// Create a follow camera
	FollowCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FollowCamera"));
	FollowCamera->SetupAttachment(CameraBoom, USpringArmComponent::SocketName); // Attach the camera to the end of the boom and let the boom adjust to match the controller orientation
	FollowCamera->bUsePawnControlRotation = false; // Camera does not rotate relative to arm

	// Note: The skeletal mesh and anim blueprint references on the Mesh component (inherited from Character) 
	// are set in the derived blueprint asset named MyCharacter (to avoid direct content references in C++)

	CurrentState.Name = TEXT("Current State");
	CurrentState.StateValues.Push(0.0);

	CurrentAction.Name = TEXT("Current Action");
	CurrentAction.ActionValues.Push(0.0);
}

//////////////////////////////////////////////////////////////////////////
// Input

void AMLR_BP2Character::SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent)
{
	// Set up gameplay key bindings
	check(PlayerInputComponent);
	PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &ACharacter::Jump);
	PlayerInputComponent->BindAction("Jump", IE_Released, this, &ACharacter::StopJumping);

	PlayerInputComponent->BindAxis("MoveForward", this, &AMLR_BP2Character::MoveForward);
	PlayerInputComponent->BindAxis("MoveRight", this, &AMLR_BP2Character::MoveRight);

	// We have 2 versions of the rotation bindings to handle different kinds of devices differently
	// "turn" handles devices that provide an absolute delta, such as a mouse.
	// "turnrate" is for devices that we choose to treat as a rate of change, such as an analog joystick
	PlayerInputComponent->BindAxis("Turn", this, &APawn::AddControllerYawInput);
	PlayerInputComponent->BindAxis("TurnRate", this, &AMLR_BP2Character::TurnAtRate);
	PlayerInputComponent->BindAxis("LookUp", this, &APawn::AddControllerPitchInput);
	PlayerInputComponent->BindAxis("LookUpRate", this, &AMLR_BP2Character::LookUpAtRate);

	// handle touch devices
	PlayerInputComponent->BindTouch(IE_Pressed, this, &AMLR_BP2Character::TouchStarted);
	PlayerInputComponent->BindTouch(IE_Released, this, &AMLR_BP2Character::TouchStopped);

	// VR headset functionality
	PlayerInputComponent->BindAction("ResetVR", IE_Pressed, this, &AMLR_BP2Character::OnResetVR);
}


void AMLR_BP2Character::OnResetVR()
{
	UHeadMountedDisplayFunctionLibrary::ResetOrientationAndPosition();
}

void AMLR_BP2Character::TouchStarted(ETouchIndex::Type FingerIndex, FVector Location)
{
		Jump();
}

void AMLR_BP2Character::TouchStopped(ETouchIndex::Type FingerIndex, FVector Location)
{
		StopJumping();
}

void AMLR_BP2Character::TurnAtRate(float Rate)
{
	// calculate delta for this frame from the rate information
	AddControllerYawInput(Rate * BaseTurnRate * GetWorld()->GetDeltaSeconds());
}

void AMLR_BP2Character::LookUpAtRate(float Rate)
{
	// calculate delta for this frame from the rate information
	AddControllerPitchInput(Rate * BaseLookUpRate * GetWorld()->GetDeltaSeconds());
}

void AMLR_BP2Character::MoveForward(float Value)
{
	if ((Controller != NULL) && (Value != 0.0f))
	{
		// find out which way is forward
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);

		// get forward vector
		const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
		AddMovementInput(Direction, Value);
	}
}

void AMLR_BP2Character::MoveRight(float Value)
{
	if ( (Controller != NULL) && (Value != 0.0f) )
	{
		// find out which way is right
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);
	
		// get right vector 
		const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
		// add movement in that direction
		AddMovementInput(Direction, Value);
	}
}

void AMLR_BP2Character::BeginPlay() {
	Super::BeginPlay();
	FVector myLoc = GetActorLocation();
	this->PrintStateStructName();
}

UFUNCTION(BlueprintCallable)
TArray<FVector> AMLR_BP2Character::GetState() {
	FVector myLoc = GetActorLocation();
	TArray<FVector> ReturnArray;
	ReturnArray.Push(myLoc);
	ReturnArray.Push(myLoc);
	ReturnArray.Push(myLoc);
	return(ReturnArray);
}

UFUNCTION(BlueprintCallable)
void AMLR_BP2Character::StateToStruct(TArray<FVector> State) {
	TArray<float> StateValues;
	for (int i = 0; i < State.Num(); ++i) {
		StateValues.Push(State[i].X);
		StateValues.Push(State[i].Y);
		StateValues.Push(State[i].Z);
	}
	CurrentState.StateValues = StateValues;
}

UFUNCTION(BlueprintCallable)
FStateStruct AMLR_BP2Character::GetStateStruct() {
	return(CurrentState);
}


UFUNCTION(BlueprintCallable)
void AMLR_BP2Character::PrintStateStructName() {
	GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, CurrentState.Name);
}

UFUNCTION(BlueprintCallable)
void AMLR_BP2Character::JsonStringToActionStruct(FString JsonString) {
	FJsonObjectConverter converter;
	bool worked = converter.JsonObjectStringToUStruct(JsonString, &CurrentAction, 0, 0);
	if (!worked) {
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, TEXT("didn't work!!"));
		return;
	}
	GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, TEXT("worked!"));
	return;
}