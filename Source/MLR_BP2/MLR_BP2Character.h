// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "MLR_BP2Character.generated.h"

USTRUCT(BlueprintType)
struct FStateStruct
{
	GENERATED_USTRUCT_BODY()

		FStateStruct() {};

		UPROPERTY()
		TArray<float> StateValues;

	UPROPERTY()
		FString Name;
};

USTRUCT(BlueprintType)
struct FActionStruct
{
	GENERATED_USTRUCT_BODY()

		FActionStruct() {};

	UPROPERTY()
		TArray<float> ActionValues;

	UPROPERTY()
		FString Name;
};

UCLASS(config=Game)
class AMLR_BP2Character : public ACharacter
{
	GENERATED_BODY()

	/** Camera boom positioning the camera behind the character */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera, meta = (AllowPrivateAccess = "true"))
	class USpringArmComponent* CameraBoom;

	/** Follow camera */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera, meta = (AllowPrivateAccess = "true"))
	class UCameraComponent* FollowCamera;
public:
	AMLR_BP2Character();

	/** Base turn rate, in deg/sec. Other scaling may affect final turn rate. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category=Camera)
	float BaseTurnRate;

	/** Base look up/down rate, in deg/sec. Other scaling may affect final rate. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category=Camera)
	float BaseLookUpRate;

	void BeginPlay();

	UFUNCTION(BlueprintCallable)
	TArray<FVector> GetState();

	UFUNCTION(BlueprintCallable)
	void StateToStruct(TArray<FVector> State);

	UPROPERTY(VisibleAnywhere)
	FStateStruct CurrentState;

	UPROPERTY(VisibleAnywhere)
	FActionStruct CurrentAction;

	UFUNCTION(BlueprintCallable)
	void PrintStateStructName();

	UFUNCTION(BlueprintCallable)
	FStateStruct GetStateStruct();

	UFUNCTION(BlueprintCallable)
	void JsonStringToActionStruct(FString JsonString);

protected:

	/** Resets HMD orientation in VR. */
	void OnResetVR();

	/** Called for forwards/backward input */
	void MoveForward(float Value);

	/** Called for side to side input */
	void MoveRight(float Value);

	/** 
	 * Called via input to turn at a given rate. 
	 * @param Rate	This is a normalized rate, i.e. 1.0 means 100% of desired turn rate
	 */
	void TurnAtRate(float Rate);

	/**
	 * Called via input to turn look up/down at a given rate. 
	 * @param Rate	This is a normalized rate, i.e. 1.0 means 100% of desired turn rate
	 */
	void LookUpAtRate(float Rate);

	/** Handler for when a touch input begins. */
	void TouchStarted(ETouchIndex::Type FingerIndex, FVector Location);

	/** Handler for when a touch input stops. */
	void TouchStopped(ETouchIndex::Type FingerIndex, FVector Location);

protected:
	// APawn interface
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;
	// End of APawn interface

public:
	/** Returns CameraBoom subobject **/
	FORCEINLINE class USpringArmComponent* GetCameraBoom() const { return CameraBoom; }
	/** Returns FollowCamera subobject **/
	FORCEINLINE class UCameraComponent* GetFollowCamera() const { return FollowCamera; }
};

