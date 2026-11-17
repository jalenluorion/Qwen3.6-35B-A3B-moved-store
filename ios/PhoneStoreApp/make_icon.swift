// The app icon: a phone with a spark, flat colours (2026-11-07). Writes the asset catalog's 1024 px icon and the loose
// iOS icon files.   swift ios/PhoneStoreApp/make_icon.swift   (from phone-store/)
import AppKit
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

func draw(_ px: Int) -> CGImage {
    let s = CGFloat(px)
    let ctx = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    ctx.setFillColor(CGColor(srgbRed: 0.16, green: 0.32, blue: 0.90, alpha: 1))   // flat blue
    ctx.fill(CGRect(x: 0, y: 0, width: s, height: s))
    // the phone: a white rounded rectangle with a darker screen
    let pw = s * 0.40, ph = s * 0.66
    let phone = CGRect(x: (s - pw) / 2 - s * 0.04, y: (s - ph) / 2, width: pw, height: ph)
    ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
    ctx.addPath(CGPath(roundedRect: phone, cornerWidth: s * 0.075, cornerHeight: s * 0.075, transform: nil))
    ctx.fillPath()
    let screen = phone.insetBy(dx: s * 0.025, dy: s * 0.025)
    ctx.setFillColor(CGColor(srgbRed: 0.10, green: 0.17, blue: 0.45, alpha: 1))
    ctx.addPath(CGPath(roundedRect: screen, cornerWidth: s * 0.055, cornerHeight: s * 0.055, transform: nil))
    ctx.fillPath()
    // the spark: a four-pointed star in warm yellow, over the phone's top-right corner
    func star(cx: CGFloat, cy: CGFloat, r: CGFloat) {
        let w = r * 0.28
        let p = CGMutablePath()
        p.move(to: CGPoint(x: cx, y: cy + r))
        p.addQuadCurve(to: CGPoint(x: cx + r, y: cy), control: CGPoint(x: cx + w, y: cy + w))
        p.addQuadCurve(to: CGPoint(x: cx, y: cy - r), control: CGPoint(x: cx + w, y: cy - w))
        p.addQuadCurve(to: CGPoint(x: cx - r, y: cy), control: CGPoint(x: cx - w, y: cy - w))
        p.addQuadCurve(to: CGPoint(x: cx, y: cy + r), control: CGPoint(x: cx - w, y: cy + w))
        ctx.addPath(p)
        ctx.fillPath()
    }
    ctx.setFillColor(CGColor(srgbRed: 1.0, green: 0.80, blue: 0.20, alpha: 1))
    star(cx: phone.maxX - s * 0.02, cy: phone.maxY - s * 0.06, r: s * 0.17)
    star(cx: phone.midX - s * 0.02, cy: phone.midY - s * 0.02, r: s * 0.09)
    return ctx.makeImage()!
}

func write(_ img: CGImage, _ path: String) {
    let d = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(d, img, nil)
    CGImageDestinationFinalize(d)
    print(path)
}

let base = "ios/PhoneStoreApp/PhoneStoreApp"
write(draw(1024), "\(base)/Assets.xcassets/AppIcon.appiconset/AppIcon-1024.png")
write(draw(120), "\(base)/Icons/AppIcon60x60@2x.png")
write(draw(180), "\(base)/Icons/AppIcon60x60@3x.png")
