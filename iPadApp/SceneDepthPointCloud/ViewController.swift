/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Main view controller for the AR experience.
*/

import UIKit
import Metal
import MetalKit
import ARKit

struct PointcloudData: Codable {
    let points: [simd_float3]
}

struct apiShapeSpaceCoeffs: Codable {
    let upper_coeffs: [Float]
    let lower_coeffs: [Float]
}

struct apiShapeSpaceCoeffsResponse: Codable {
    let upper_coeffs: [Float]
    let lower_coeffs: [Float]
    let blade_start_x: Float
    let blade_start_y: Float
    let blade_end_x: Float
    let blade_end_y: Float
    let aerofoil_mesh: [[Float]]
}

struct apiCfdPoint: Codable {
    let x: Float
    let y: Float
    let i: Int32
    let j: Int32
    let pstat: Float
    let pstag: Float
    let vx: Float
    let vt: Float
}

struct apiCfdResponse: Codable {
    let pitch: Float
    let pstat_min: Float
    let pstat_max: Float
    let pstag_min: Float
    let pstag_max: Float
    let vx_min: Float
    let vx_max: Float
    let vt_min: Float
    let vt_max: Float
    let indices: [Int32]
    let points: [apiCfdPoint]
}

final class ViewController: UIViewController, ARSessionDelegate, UIPickerViewDelegate, UIPickerViewDataSource {
    
    @IBOutlet weak var toolbar: UIToolbar!
    @IBOutlet weak var statusLabel: UIBarButtonItem!
    @IBOutlet weak var rgbRadiusSlider: UISlider!
    @IBOutlet weak var resetButton: UIBarButtonItem!
    @IBOutlet weak var spinIndicator: UIActivityIndicatorView!
    @IBOutlet weak var aerofoilSidebar: UIView!
    @IBOutlet weak var pitchValueLabel: UILabel!
    @IBOutlet weak var cfdSidebar: UIView!
    @IBOutlet weak var meshPicker: UIPickerView!
    @IBOutlet weak var minLabel: UILabel!
    @IBOutlet weak var maxLabel: UILabel!
    
    private let session = ARSession()
    private var renderer: Renderer!
    
    private let serverAddr = "192.168.68.117"
    private var markerAnchor: ARImageAnchor!
    
    var upper_coeffs: [Float] = []
    var lower_coeffs: [Float] = []
    
    var meshPickerData: [String] = [String]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        session.delegate = self
        
        // Set the view to use the default device
        if let view = view as? MTKView {
            view.device = device
            
            view.backgroundColor = UIColor.clear
            // we need this to enable depth test
            view.depthStencilPixelFormat = .depth32Float
            view.contentScaleFactor = 1
            view.delegate = self
            
            // Configure the renderer to draw to the view
            renderer = Renderer(session: session, metalDevice: device, renderDestination: view)
            renderer.drawRectResized(size: view.bounds.size)
        }
        
        meshPicker.delegate = self
        meshPicker.dataSource = self
        
        meshPickerData = ["Static Pressure", "Stagnation Pressure", "Vx", "Vt"]
        
        statusLabel.title = "Status: Finding marker"
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        guard let referenceImages = ARReferenceImage.referenceImages(inGroupNamed: "AR Resources", bundle: nil) else {
            fatalError("Missing expected asset catalog resources.")
        }
        
        // Create a world-tracking configuration, and
        // enable the scene depth frame-semantic.
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = .sceneDepth
        configuration.detectionImages = referenceImages

        // Run the view's session
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        
        // The screen shouldn't dim during AR experiences.
        UIApplication.shared.isIdleTimerDisabled = true
    }
    
    // Auto-hide the home indicator to maximize immersion in AR experiences.
    override var prefersHomeIndicatorAutoHidden: Bool {
        return true
    }
    
    // Hide the status bar to maximize immersion in AR experiences.
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user.
        guard error is ARError else { return }
        let errorWithInfo = error as NSError
        let messages = [
            errorWithInfo.localizedDescription,
            errorWithInfo.localizedFailureReason,
            errorWithInfo.localizedRecoverySuggestion
        ]
        let errorMessage = messages.compactMap({ $0 }).joined(separator: "\n")
        DispatchQueue.main.async {
            // Present an alert informing about the error that has occurred.
            let alertController = UIAlertController(title: "The AR session failed.", message: errorMessage, preferredStyle: .alert)
            let restartAction = UIAlertAction(title: "Restart Session", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
                if let configuration = self.session.configuration {
                    self.session.run(configuration, options: .resetSceneReconstruction)
                }
            }
            alertController.addAction(restartAction)
            self.present(alertController, animated: true, completion: nil)
        }
    }
    
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        guard let imageAnchor = anchors[0] as? ARImageAnchor else { return }
        print(imageAnchor)

        let normalizeTransform = normalize(transform: imageAnchor.transform)
        session.setWorldOrigin(relativeTransform: normalizeTransform)
        
        self.markerAnchor = imageAnchor
        renderer.markerFound = true
        renderer.showPointcloud = true
        renderer.rgbRadius = 0.5
        rgbRadiusSlider.value = 0.5
        statusLabel.title = "Status: Scanning aerofoil"
        resetButton.isEnabled = true
    }
    
    func normalize(transform: simd_float4x4) -> simd_float4x4 {
        var (c1, c2, c3, c4) = transform.columns

        c1 = simd_normalize(c1)
        c2 = simd_normalize(c2)
        c3 = simd_normalize(c3)
        
        return simd_float4x4(c1,c2,c3,c4)
    }
    
    @IBAction func saveButtonPress(_ sender: UIBarButtonItem) {
        print("save button pressed")
        
        if !renderer.aerofoilMeshReady {
            let pointcloud = PointcloudData(points: renderer.exportPointcloud())
            
            guard let uploadData = try? JSONEncoder().encode(pointcloud) else {
                return
            }
            
            let url = URL(string: "http://\(serverAddr):8080/pointcloud")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            let task = URLSession.shared.uploadTask(with: request, from: uploadData) { data, response, error in
                if let error = error {
                    print("error: \(error)")
                    return
                }
                guard let response = response as? HTTPURLResponse,
                    (200...299).contains(response.statusCode) else {
                    print("server error")
                    return
                }
                
                print("got response!")

                guard let data = data, let responseData = try? JSONDecoder().decode(apiShapeSpaceCoeffsResponse.self, from: data) else {
                    print("Cannot decode API data.")
                    return
                }
                
                print("got upper coeffs \(responseData.upper_coeffs)")
                print("got lower coeffs \(responseData.lower_coeffs)")
                
                self.upper_coeffs = responseData.upper_coeffs
                self.lower_coeffs = responseData.lower_coeffs
                self.renderer.setAerofoilMesh(responseData: responseData)
                
                DispatchQueue.main.async {
                    self.aerofoilSidebar.isHidden = false
                    self.spinIndicator.stopAnimating()
                    self.statusLabel.title = "Status: Confirm blade properties"
                }
            }
            task.resume()
            
            statusLabel.title = "Status: Processing pointcloud"
            renderer.showPointcloud = false
            renderer.rgbRadius = 1
            rgbRadiusSlider.value = 1
            spinIndicator.startAnimating()
        } else {
            let coeffs = apiShapeSpaceCoeffs(upper_coeffs: upper_coeffs, lower_coeffs: lower_coeffs)
            
            guard let uploadData = try? JSONEncoder().encode(coeffs) else {
                return
            }
            
            let url = URL(string: "http://\(serverAddr):8080/cfd")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            let task = URLSession.shared.uploadTask(with: request, from: uploadData) { data, response, error in
                if let error = error {
                    print("error: \(error)")
                    return
                }
                guard let response = response as? HTTPURLResponse,
                    (200...299).contains(response.statusCode) else {
                    print("server error")
                    return
                }
                
                print("got response!")

                guard let data = data, let responseData = try? JSONDecoder().decode(apiCfdResponse.self, from: data) else {
                    print("Cannot decode API data.")
                    return
                }
                
                print("decoded \(responseData.points.count) points.")
                print("decoded \(responseData.indices.count) indices.")
                self.renderer.setCfdResults(responseData: responseData)
                
                DispatchQueue.main.async {
                    self.aerofoilSidebar.isHidden = true
                    self.cfdSidebar.isHidden = false
                    self.toolbar.isHidden = true
                    self.spinIndicator.stopAnimating()

                    self.minLabel.text = String(format: "%.2f", self.renderer.cfdUniformsBuffer[0].pstatMin)
                    self.maxLabel.text = String(format: "%.2f", self.renderer.cfdUniformsBuffer[0].pstatMax)
                    self.minLabel.isHidden = false
                    self.maxLabel.isHidden = false
                }
            }
            task.resume()
            
            statusLabel.title = "Status: Running CFD"
            spinIndicator.startAnimating()
        }
    }
    
    @IBAction func rgbSliderChanged(_ sender: UISlider) {
        renderer.rgbRadius = sender.value
    }
    
    
    @IBAction func resetPressed(_ sender: UIBarButtonItem) {
        if (self.markerAnchor == nil) {
            return
        }
        
        session.remove(anchor: self.markerAnchor)
        renderer.markerFound = false
        renderer.aerofoilMeshReady = false
    
        let newBuffer = [ParticleUniforms](repeating: ParticleUniforms(), count: renderer.particlesBuffer.count)
        renderer.particlesBuffer.assign(with: newBuffer)
        
        resetButton.isEnabled = false
        renderer.rgbRadius = 1.5
        rgbRadiusSlider.value = 1.5
        aerofoilSidebar.isHidden = true
        
        statusLabel.title = "Status: Finding marker"
    }
    
    
    @IBAction func pitchSliderChanged(_ sender: UISlider) {
        renderer.cfdUniformsBuffer[0].pitch = sender.value / 1000
        pitchValueLabel.text = "\(String(format: "%.1f", sender.value)) mm"
    }
    
    @IBAction func gridlineSwitchChanged(_ sender: UISwitch) {
        renderer.cfdUniformsBuffer[0].showGridlines = sender.isOn ? Int32(1) : Int32(0)
    }

    // Number of columns of data
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    // The number of rows of data
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return meshPickerData.count
    }
    
    // The data to return for the row and component (column) that's being passed in
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return meshPickerData[row]
    }
    
    // Capture the picker view selection
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        // This method is triggered whenever the user makes a change to the picker selection.
        // The parameter named row and component represents what was selected.
        renderer.cfdUniformsBuffer[0].meshType = Int32(row)
        
        switch row {
        case 0:
            minLabel.text = String(format: "%.2f", renderer.cfdUniformsBuffer[0].pstatMin)
            maxLabel.text = String(format: "%.2f", renderer.cfdUniformsBuffer[0].pstatMax)
        case 1:
            minLabel.text = String(format: "%.2f", renderer.cfdUniformsBuffer[0].pstagMin)
            maxLabel.text = String(format: "%.2f", renderer.cfdUniformsBuffer[0].pstagMax)
        case 2:
            minLabel.text = String(format: "%.2f", renderer.cfdUniformsBuffer[0].vxMin)
            maxLabel.text = String(format: "%.2f", renderer.cfdUniformsBuffer[0].vxMax)
        case 3:
            minLabel.text = String(format: "%.2f", renderer.cfdUniformsBuffer[0].vtMin)
            maxLabel.text = String(format: "%.2f", renderer.cfdUniformsBuffer[0].vtMax)
        default:
            break
        }
    }
}

// MARK: - MTKViewDelegate

extension ViewController: MTKViewDelegate {
    // Called whenever view changes orientation or layout is changed
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.drawRectResized(size: size)
    }
    
    // Called whenever the view needs to render
    func draw(in view: MTKView) {
        renderer.draw()
    }
}

// MARK: - RenderDestinationProvider

protocol RenderDestinationProvider {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? { get }
    var currentDrawable: CAMetalDrawable? { get }
    var colorPixelFormat: MTLPixelFormat { get set }
    var depthStencilPixelFormat: MTLPixelFormat { get set }
    var sampleCount: Int { get set }
}

extension MTKView: RenderDestinationProvider {
    
}
