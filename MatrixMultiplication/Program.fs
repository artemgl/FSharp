module MatrixMultiply

open OpenCL.Net
open Brahma.OpenCL
open Brahma.FSharp.OpenCL.Core
open Brahma.FSharp.OpenCL.Extensions

let random = new System.Random()
       
let MakeMatrix rows cols =
   Array.init (rows * cols) (fun i -> float32 (random.NextDouble()))
   
let Main platformName fSize sSize tSize =

   let m1 = (MakeMatrix fSize sSize)
   let m2 = (MakeMatrix sSize tSize)
   let localWorkSize = 2
   let deviceType = DeviceType.Default

   let provider =
       try  ComputeProvider.Create(platformName, deviceType)
       with 
       | ex -> failwith ex.Message

   let mutable commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head)

   let aValues = m1
   let bValues = m2
   let cParallel = Array.zeroCreate(fSize * tSize)

   let command = 
       <@
           fun (r:_2D) (a:array<_>) (b:array<_>) (c:array<_>) -> 
               let tx = r.GlobalID0
               let ty = r.GlobalID1
               let mutable buf = c.[ty * tSize + tx]
               for k in 0 .. sSize - 1 do
                   buf <- buf + (a.[ty * fSize + k] * b.[k * sSize + tx])
               c.[ty * tSize + tx] <- buf
       @>

   let kernel, kernelPrepare, kernelRun = provider.Compile command
   let d =(new _2D(fSize, tSize, localWorkSize, localWorkSize))
   kernelPrepare d aValues bValues cParallel

   commandQueue.Add(kernelRun()).Finish() |> ignore

   let _ = commandQueue.Add(cParallel.ToHost provider).Finish()

   commandQueue.Dispose()
   provider.CloseAllBuffers()
   provider.Dispose()
           
Main "NVIDIA*" 4 4 4