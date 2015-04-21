//
// Cuda.Learn - http://github.com/redknightlois/cudalearn
// Copyright (c) Corvalius 
//
// FAKE build script, see http://fsharp.github.io/FAKE
//


// include Fake lib
//#I "packages/FAKE/tools"
#r "packages/FAKE/tools/FakeLib.dll"

open Fake
open Fake.StringHelper
open System
 
Environment.CurrentDirectory <- __SOURCE_DIRECTORY__
let header = ReadFile(__SOURCE_DIRECTORY__ @@ "build.fsx") |> Seq.take 6 |> Seq.map (fun s -> s.Substring(2)) |> toLines
trace header

// Properties
let buildDir = "./build/"
let deployDir = "./deploy/"

// --------------------------------------------------------------------------------------
// PREPARE
// --------------------------------------------------------------------------------------

RestorePackages()

Target "Start" DoNothing
 
// Targets
Target "Clean" (fun _ ->
    CleanDirs [buildDir; deployDir]
)

Target "Prepare" DoNothing

"Start"
  =?> ("Clean", not (hasBuildParam "incremental"))
  ==> "Prepare"

// --------------------------------------------------------------------------------------
// BUILD
// --------------------------------------------------------------------------------------

let buildMode = getBuildParamOrDefault "buildMode" "Release"
let properties = [("Configuration", buildMode ); ("Platform","x64"); ("Optimize", "True"); ("DebugSymbols", "True");]
let build subject = MSBuild buildDir (if hasBuildParam "incremental" then "Build" else "Rebuild") properties subject |> ignore

Target "Build" (fun _ -> build !! "CudaLearn.sln")

// start build
RunTargetOrDefault "Build"