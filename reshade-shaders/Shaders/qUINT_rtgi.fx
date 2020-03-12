/*=============================================================================

	ReShade 4 effect file
    github.com/martymcmodding

	Support me:
   		paypal.me/mcflypg
   		patreon.com/mcflypg

    Path Traced Global Illumination 

    * Unauthorized copying of this file, via any medium is strictly prohibited
 	* Proprietary and confidential

=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

#ifndef INFINITE_BOUNCES
 #define INFINITE_BOUNCES       0   //[0 or 1]      If enabled, path tracer samples previous frame GI as well, causing a feedback loop to simulate secondary bounces, causing a more widespread GI.
#endif

#ifndef SKYCOLOR_MODE
 #define SKYCOLOR_MODE          0   //[0 to 2]      0: skycolor feature disabled | 1: manual skycolor | 2: dynamic skycolor
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform float RT_SIZE_SCALE <
	ui_type = "drag";
	ui_min = 0.5; ui_max = 1.0;
    ui_step = 0.5;
    ui_label = "GI Render Resolution Scale";
    ui_category = "Global";
> = 1.0;

uniform float RT_SAMPLE_RADIUS <
	ui_type = "drag";
	ui_min = 0.5; ui_max = 20.0;
    ui_step = 0.01;
    ui_label = "Ray Length";
	ui_tooltip = "Maximum ray length, directly affects\nthe spread radius of shadows / indirect lighting";
    ui_category = "Path Tracing";
> = 4.0;

uniform int RT_RAY_AMOUNT <
	ui_type = "slider";
	ui_min = 1; ui_max = 20;
    ui_label = "Ray Amount";
    ui_category = "Path Tracing";
> = 3;

uniform int RT_RAY_STEPS <
	ui_type = "slider";
	ui_min = 1; ui_max = 20;
    ui_label = "Ray Step Amount";
    ui_category = "Path Tracing";
> = 12;

uniform bool RT_HIGHP_LIGHT_SPREAD <
    ui_label = "Enable precise light spreading";
    ui_category = "Path Tracing";
> = true;

uniform float RT_Z_THICKNESS <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 4.0;
    ui_step = 0.01;
    ui_label = "Z Thickness";
	ui_tooltip = "The shader can't know how thick objects are, since it only\nsees the side the camera faces and has to assume a fixed value.\n\nUse this parameter to remove halos around thin objects.";
    ui_category = "Path Tracing";
> = 0.5;

#if SKYCOLOR_MODE != 0

#if SKYCOLOR_MODE == 1
uniform float3 SKY_COLOR <
	ui_type = "color";
	ui_label = "Sky Color";
    ui_category = "Blending";
> = float3(1.0, 0.0, 0.0);
#endif

#if SKYCOLOR_MODE == 2
uniform float SKY_COLOR_SAT <
	ui_type = "drag";
	ui_min = 0; ui_max = 5.0;
    ui_step = 0.01;
    ui_label = "Auto Sky Color Saturation";
    ui_category = "Blending";
> = 1.0;
#endif

uniform float SKY_COLOR_AMBIENT_MIX <
	ui_type = "drag";
	ui_min = 0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Sky Color Ambient Mix";
    ui_tooltip = "How much of the occluded ambient color is considered skycolor\n\nIf 0, Ambient Occlusion removes white ambient color,\nif 1, Ambient Occlusion only removes skycolor";
    ui_category = "Blending";
> = 0.2;

uniform float SKY_COLOR_AMT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Sky Color Intensity";
    ui_category = "Blending";
> = 4.0;
#endif

uniform float RT_AO_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Ambient Occlusion Intensity";
    ui_category = "Blending";
> = 4.0;

uniform float RT_IL_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Bounce Lighting Intensity";
    ui_category = "Blending";
> = 4.0;

#if INFINITE_BOUNCES != 0
    uniform float RT_IL_BOUNCE_WEIGHT <
        ui_type = "drag";
        ui_min = 0; ui_max = 5.0;
        ui_step = 0.01;
        ui_label = "Next Bounce Weight";
        ui_category = "Blending";
    > = 0.0;
#endif

uniform float2 RT_FADE_DEPTH <
	ui_type = "drag";
    ui_label = "Fade Out Start / End";
	ui_min = 0.00; ui_max = 1.00;
	ui_tooltip = "Distance where GI starts to fade out | is completely faded out.";
    ui_category = "Blending";
> = float2(0.0, 0.5);

uniform int RT_DEBUG_VIEW <
	ui_type = "radio";
    ui_label = "Enable Debug View";
	ui_items = "None\0Lighting Channel\0";
	ui_tooltip = "Different debug outputs";
    ui_category = "Debug";
> = 0;

/*
uniform float4 tempF1 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF2 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF3 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);
*/

/*=============================================================================
	Textures, Samplers, Globals
=============================================================================*/

#define RESHADE_QUINT_COMMON_VERSION_REQUIRE 200
#include "qUINT_common.fxh"

#define CONST_LOG2(v) (((v) & 0xAAAAAAAA) != 0) | ((((v) & 0xFFFF0000) != 0) << 4) | ((((v) & 0xFF00FF00) != 0) << 3) | ((((v) & 0xF0F0F0F0) != 0) << 2) | ((((v) & 0xCCCCCCCC) != 0) << 1)

//for 1920x1080, use 3 mip levels
//double the screen size, use one mip level more
//log2(1920/240) = 3
//log2(3840/240) = 4
#define MIP_AMT 	CONST_LOG2(BUFFER_WIDTH / 240)
#define MIP_BIAS_IL 2

texture ZTex 	           < pooled = true; > 	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F;      MipLevels = MIP_AMT;};
texture NormalTex 	       < pooled = true; > 	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGB10A2;   };
texture ColorTex 	       < pooled = true; > 	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGB10A2;   MipLevels = MIP_AMT + MIP_BIAS_IL;  };
texture GITex	            					{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITexPrev	       						{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITexTemp	        					{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GBufferTexPrev      					{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture HistoryConfidence  < pooled = true; > 	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R8; };

texture SkyCol          { Width = 1;   Height = 1;   Format = RGBA8; };
texture SkyColPrev      { Width = 1;   Height = 1;   Format = RGBA8; };
sampler2D sSkyCol	    { Texture = SkyCol;	};
sampler2D sSkyColPrev	{ Texture = SkyColPrev;	};

texture JitterTex           < source = "bluenoise.png"; > { Width = 32; Height = 32; Format = RGBA8; };

sampler2D sZTex	            { Texture = ZTex;	    };
sampler2D sNormalTex	    { Texture = NormalTex;	};
sampler2D sColorTex	        { Texture = ColorTex;	};
sampler2D sGITex	        { Texture = GITex;	        };
sampler2D sGITexPrev	    { Texture = GITexPrev;	    };
sampler2D sGITexTemp	    { Texture = GITexTemp;	    };
sampler2D sGBufferTexPrev	{ Texture = GBufferTexPrev;	};
sampler2D sHistoryConfidence	{ Texture = HistoryConfidence;	};

sampler	sJitterTex          { Texture = JitterTex; AddressU = WRAP; AddressV = WRAP;};

/*=============================================================================
	Vertex Shader
=============================================================================*/

struct VSOUT
{
	float4                  vpos        : SV_Position;
    float2                  uv          : TEXCOORD0;
    float4                  uv_scaled   : TEXCOORD1;
};

VSOUT VS_RT(in uint id : SV_VertexID)
{
    VSOUT o;

    o.uv.x = (id == 2) ? 2.0 : 0.0;
    o.uv.y = (id == 1) ? 2.0 : 0.0;

    o.uv_scaled.xy = o.uv / RT_SIZE_SCALE;
    o.uv_scaled.zw = o.uv * RT_SIZE_SCALE;

    o.vpos = float4(o.uv.xy * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

/*=============================================================================
	Functions
=============================================================================*/

struct MRT
{
    float4 gi   : SV_Target0;
    float4 gbuf : SV_Target1;
};

struct RTConstants
{
    float3 pos;
    float3 normal;
    int nrays;
    int nsteps;
};

#include "RTGI/Projection.fxh"
#include "RTGI/Normal.fxh"
#include "RTGI/RaySorting.fxh"
#include "RTGI/RayTracing.fxh"

void unpack_hdr(inout float3 color)
{
    color = color * rcp(1.01 - saturate(color));    
}

void pack_hdr(inout float3 color)
{
    color = 1.01 * color * rcp(color + 1.0);
}

float3 dither(in VSOUT i)
{
    const float2 magicdot = float2(0.75487766624669276, 0.569840290998);
    const float3 magicadd = float3(0, 0.025, 0.0125) * dot(magicdot, 1);

    const int bit_depth = 8; //TODO: add BUFFER_COLOR_DEPTH once it works
    const float lsb = exp2(bit_depth) - 1;

    float3 dither = frac(dot(i.vpos.xy, magicdot) + magicadd);
    dither /= lsb;
    
    return dither;
}

float compute_history_confidence(MRT curr, MRT prev, inout float historyconfidence)
{
    float4 gbuf_delta = abs(curr.gbuf - prev.gbuf);

    historyconfidence = dot(gbuf_delta.xyz, gbuf_delta.xyz) * 10 + gbuf_delta.w;
    historyconfidence = exp(-historyconfidence);

    return lerp(0.25, 0.9, saturate(1 - historyconfidence));
}

float4 fetch_gbuffer(in float2 uv)
{
    return float4(tex2Dlod(sNormalTex, float4(uv, 0, 0)).xyz * 2 - 1, 
                  tex2Dlod(sZTex, 	   float4(uv, 0, 0)).x);
}

float4 atrous(int iter, sampler gi, VSOUT i)
{
    float4 center_gbuf = fetch_gbuffer(i.uv);
    float4 center_gi = tex2D(gi, i.uv);

    float historyconfidence = tex2D(sHistoryConfidence, i.uv_scaled.zw).x;

    float4 weighted_value_sum = 0;
    float weight_sum = 0.00001;

    float size[4] = {1.5, 3, 6, 12};
    float error_thresh_val[4] = {0.85, 3.5, 32, 64};

    [unroll]for(int x = -1; x <= 1; x++)
    [unroll]for(int y = -1; y <= 1; y++)
    {
        float2 grid_pos = float2(x, y) * size[iter] * qUINT::PIXEL_SIZE;

        float2 tap_uv       = i.uv + grid_pos;  
        float4 tap_gi       = tex2Dlod(gi, float4(tap_uv, 0, 0));
        float4 tap_gbuf     = fetch_gbuffer(tap_uv);

        float wz = 16.0 *  (1.0 - tap_gbuf.w / center_gbuf.w);
        wz = saturate(0.5 - lerp(wz, abs(wz), 0.75)); //bias away from camera
        float wn = saturate(dot(tap_gbuf.xyz, center_gbuf.xyz) * 1.6 - 0.6); 
        float wi = dot(abs(tap_gi - center_gi), float4(0.9, 1.77, 0.33, 3.0));
        wi = exp(-wi*wi*error_thresh_val[iter]);

        wn = lerp(wn, 1, saturate(wz * 1.42 - 0.42)); //adjust n if z is very close
        wi = lerp(wi, 1, saturate(1 - historyconfidence  * 1.2)); //adjust value to counteract disocclusion noise

        float w = saturate(wz * wn * wi);

        weighted_value_sum += tap_gi * w;
        weight_sum += w;
    }

    //return center_gi;
    return weighted_value_sum / weight_sum;
}

/*=============================================================================
	Pixel Shaders
=============================================================================*/

void PS_Deferred(in VSOUT i, out float4 color : SV_Target0, out float4 normal : SV_Target1, out float depth : SV_Target2)
{	
    color 	= tex2D(qUINT::sBackBufferTex, i.uv);
    normal  = float4(Normal::normal_from_depth(i) * 0.5 + 0.5, 1); 
    depth   = Projection::depth_to_z(qUINT::linear_depth(i.uv));
}

void PS_RTMain(in VSOUT i, out float4 o : SV_Target0, out float historyconfidence : SV_Target1)
{
    RTConstants rtconstants;
    rtconstants.pos     = Projection::uv_to_proj(i.uv_scaled.xy);
    rtconstants.normal  = tex2D(sNormalTex, i.uv_scaled.xy).xyz * 2 - 1;
    rtconstants.nrays   = RT_RAY_AMOUNT;
    rtconstants.nsteps  = RT_RAY_STEPS;  

    float depth = Projection::z_to_depth(rtconstants.pos.z); 
    rtconstants.pos = rtconstants.pos * 0.998 + rtconstants.normal * depth;

    MRT curr, prev; 

    float2 bluenoise = tex2Dfetch(sJitterTex, int4(i.vpos.xy % tex2Dsize(sJitterTex), 0, 0)).xy;
    float3x3 ray_to_hemisphere = Normal::base_from_vector(rtconstants.normal);

    SampleSet sampleset;
    ray_sorting(i, qUINT::FRAME_COUNT, bluenoise.x, sampleset);

    curr.gi = 0;

    [loop]
    for(int r = 0; r < 0 + rtconstants.nrays; r++)
    {
        RayTracing::Ray ray;
        ray.pos = rtconstants.pos;
        //generate unit ray, cosine weighted ray density (uniform: use sqrt of z)
        ray.dir.z = (r + sampleset.index) / rtconstants.nrays;
        ray.dir.xy = 1 - ray.dir.z;
        ray.dir = sqrt(ray.dir);
        ray.dir.xy *= sampleset.dir_xy;
        //reorient ray to surface alignment
        ray.dir = mul(ray.dir, ray_to_hemisphere);       
        ray.maxlen = RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS;

        //advance to next ray dir
        sampleset.dir_xy = mul(sampleset.dir_xy, sampleset.nextdir); 

        ray.steplen = ray.maxlen * rsqrt(dot(ray.dir.xy, ray.dir.xy) + 1e-3) / rtconstants.nsteps; 
        ray.currlen = ray.steplen * bluenoise.y;

        float intersected = RayTracing::compute_intersection(ray, rtconstants, i);
        curr.gi.w += intersected;

        if(RT_IL_AMOUNT * intersected == 0) 
			continue;

		float3 albedo           = tex2Dlod(sColorTex, 	float4(ray.uv, 0, ray.width + MIP_BIAS_IL)).rgb; unpack_hdr(albedo);
        float3 intersect_normal = tex2Dlod(sNormalTex,  float4(ray.uv, 0, 0)).xyz * 2.0 - 1.0;

#if INFINITE_BOUNCES != 0
        float3 nextbounce 		= tex2Dlod(sGITexPrev, 	float4(ray.uv, 0, 0)).rgb; unpack_hdr(nextbounce);            
        albedo += nextbounce * RT_IL_BOUNCE_WEIGHT;
#endif

		float light_angle = saturate(dot(-intersect_normal, ray.dir));
        curr.gi.rgb += albedo * light_angle;        
    }

    curr.gi /= rtconstants.nrays; 
    pack_hdr(curr.gi.rgb);

    curr.gbuf = float4(rtconstants.normal, rtconstants.pos.z);  
    prev.gi = tex2D(sGITexTemp, i.uv_scaled.xy);
    prev.gbuf = tex2D(sGBufferTexPrev, i.uv_scaled.xy);

    float alpha = compute_history_confidence(curr, prev, historyconfidence);
    o = lerp(prev.gi, curr.gi, alpha);
}

void PS_Copy(in VSOUT i, out MRT o)
{	
    o.gi    = tex2D(sGITex, i.uv_scaled.zw);
    o.gbuf  = fetch_gbuffer(i.uv);

    if(qUINT::linear_depth(i.uv.xy) >= max(RT_FADE_DEPTH.x, RT_FADE_DEPTH.y) //theoretically only .y but users might swap it...
    ) discard;
}

void PS_Filter0(in VSOUT i, out float4 o : SV_Target0)
{
    o = atrous(0, sGITexPrev, i);
}
void PS_Filter1(in VSOUT i, out float4 o : SV_Target0)
{
    o = atrous(1, sGITexTemp, i);
}
void PS_Filter2(in VSOUT i, out float4 o : SV_Target0)
{
    o = atrous(2, sGITex, i);
}
void PS_Filter3(in VSOUT i, out float4 o : SV_Target0)
{
    o = atrous(3, sGITexTemp, i);
}

void PS_Output(in VSOUT i, out float4 o : SV_Target0)
{
    float4 gi = tex2D(sGITex, i.uv);//sGITexTemp
    float3 color = tex2D(qUINT::sBackBufferTex, i.uv).rgb;

    unpack_hdr(color);
    unpack_hdr(gi.rgb);    

    if(RT_DEBUG_VIEW == 1) color.rgb = 1;

#if SKYCOLOR_MODE != 0
 #if SKYCOLOR_MODE == 1
    float3 skycol = SKY_COLOR;
 #else
    float3 skycol = tex2Dfetch(sSkyCol, 0).rgb;
    skycol = lerp(dot(skycol, 0.333), skycol, SKY_COLOR_SAT * 0.2);
 #endif

    float fade = smoothstep(RT_FADE_DEPTH.y, RT_FADE_DEPTH.x, qUINT::linear_depth(i.uv));   
    gi *= fade;  
    skycol *= fade;  

    color = color * (1.0 + gi.rgb * RT_IL_AMOUNT * RT_IL_AMOUNT); //apply GI
    color = color / (1.0 + lerp(1.0, skycol, SKY_COLOR_AMBIENT_MIX) * gi.w * RT_AO_AMOUNT); //apply AO as occlusion of skycolor
    color = color * (1.0 + skycol * SKY_COLOR_AMT);
#else
    float fade = smoothstep(RT_FADE_DEPTH.y, RT_FADE_DEPTH.x, qUINT::linear_depth(i.uv));   
    gi *= fade;  

    color = color * (1.0 + gi.rgb * RT_IL_AMOUNT * RT_IL_AMOUNT); //apply GI
    color = color / (1.0 + gi.w * RT_AO_AMOUNT);
#endif

    pack_hdr(color.rgb);

    //dither a little bit as large scale lighting might exhibit banding
    color.rgb += dither(i);
    o = float4(color, 1);
}

void PS_ReadSkycol(in VSOUT i, out float4 o : SV_Target0)
{
    float2 gridpos;
    gridpos.x = qUINT::FRAME_COUNT % 64;
    gridpos.y = floor(qUINT::FRAME_COUNT / 64) % 64;

    float2 unormgridpos = gridpos / 64.0;

    int searchsize = 10;

    float4 skycolor = 0.0;

    for(float x = 0; x < searchsize; x++)
    for(float y = 0; y < searchsize; y++)
    {
        float2 loc = (float2(x, y) + unormgridpos) * rcp(searchsize);

        float z = qUINT::linear_depth(loc);
        float issky = z == 1;

        skycolor += float4(tex2Dlod(qUINT::sBackBufferTex, float4(loc, 0, 0)).rgb, 1) * issky;
    }

    skycolor.rgb /= skycolor.w + 0.000001;

    float4 prevskycolor = tex2D(sSkyColPrev, 1);

    bool skydetectedthisframe = skycolor.w > 0.000001;
    bool skydetectedatall = prevskycolor.w; //0 if skycolor has not been read yet at all

    float interp = 0;

    //no skycol yet stored, now we have skycolor, use it
    if(!skydetectedatall && skydetectedthisframe)
        interp = 1;

    if(skydetectedatall && skydetectedthisframe)
        interp = saturate(0.1 * 0.01 * qUINT::FRAME_TIME);

    o.rgb = lerp(prevskycolor.rgb, skycolor.rgb, interp);
    o.w = skydetectedthisframe || skydetectedatall;
}

void PS_CopyPrevSkycol(in VSOUT i, out float4 o : SV_Target0)
{
    o = tex2D(sSkyCol, 1.0);
}

void PS_StencilSetup(in VSOUT i, out float4 o : SV_Target0)
{   
    o = tex2D(qUINT::sBackBufferTex, i.uv);

    if(qUINT::linear_depth(i.uv_scaled.xy) >= max(RT_FADE_DEPTH.x, RT_FADE_DEPTH.y) //theoretically only .y but users might swap it...
    || max(i.uv_scaled.x, i.uv_scaled.y) > 1
    ) discard;    
}

/*=============================================================================
	Techniques
=============================================================================*/

#define CREATE_STENCIL StencilEnable = true; \
                       StencilPass = REPLACE; \
                       StencilRef = 1;


#define USE_STENCIL     ClearRenderTargets = true; \
                        StencilEnable = true; \
                        StencilPass = KEEP; \
                        StencilFunc = EQUAL; \
                        StencilRef = 1;

technique RTGlobalIllumination
< ui_tooltip = "              >> qUINT::RTGI 0.9 <<\n\n"
			   "         EARLY ACCESS -- PATREON ONLY\n"
               "Official versions only via patreon.com/mcflypg\n"
               "\nRTGI is written by Marty McFly / Pascal Gilcher\n"
               "Early access, featureset might be subject to change"; >
{
#if SKYCOLOR_MODE == 2
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_ReadSkycol;
        RenderTarget = SkyCol;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_CopyPrevSkycol;
        RenderTarget = SkyColPrev;
	}
#endif
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Deferred;
        RenderTarget0 = ColorTex;
        RenderTarget1 = NormalTex;
        RenderTarget2 = ZTex;
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_StencilSetup;
        ClearRenderTargets = false;
		CREATE_STENCIL
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_RTMain;
        RenderTarget0 = GITex;
        RenderTarget1 = HistoryConfidence;
        USE_STENCIL
	}  
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Copy;
        RenderTarget0 = GITexPrev;
        RenderTarget1 = GBufferTexPrev;
        ClearRenderTargets = true;
		CREATE_STENCIL
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Filter0;
        RenderTarget = GITexTemp;
        USE_STENCIL
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Filter1;
        RenderTarget = GITex;
        USE_STENCIL
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Filter2;
        RenderTarget = GITexTemp;
        USE_STENCIL
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Filter3;
        RenderTarget = GITex;
        USE_STENCIL
	}
    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Output;
	}
}